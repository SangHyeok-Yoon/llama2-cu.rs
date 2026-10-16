// To download the model...
// wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

use std::fs::File;
use std::io::{Read, Write};
use std::ops::Range;
use std::path::Path;
use std::{fs, io};

struct Config {
    dim: i32,        // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32,   // number of layers
    n_heads: i32,    // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32,    // max sequence length
    shared_weights: bool,
}

impl Config {
    fn try_from(bytes: &[u8]) -> Result<Self, std::io::Error> {
        let vocab_size = i32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let shared_weights = vocab_size > 0;
        Ok(Self {
            dim: i32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            hidden_dim: i32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            n_layers: i32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            n_heads: i32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            n_kv_heads: i32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            vocab_size: vocab_size.abs(),
            seq_len: i32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            shared_weights,
        })
    }
}

struct TransformerWeights {
    data: Vec<f32>,

    // token embedding table
    token_embedding_table: Range<usize>, // (vocab_size, dim)

    // weights for rmsnorms
    rms_att_weight: Range<usize>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Range<usize>, // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size
    wq: Range<usize>, // (layer, dim, n_heads * head_size)
    wk: Range<usize>, // (layer, dim, n_kv_heads * head_size)
    wv: Range<usize>, // (layer, dim, n_kv_heads * head_size)
    wo: Range<usize>, // (layer, n_heads * head_size, dim)

    // weights for ffn
    w1: Range<usize>, // (layer, hidden_dim, dim)
    w2: Range<usize>, // (layer, dim, hidden_dim)
    w3: Range<usize>, // (layer, hidden_dim, dim)

    // final rmsnorm
    rms_final_weight: Range<usize>, // (dim,)

    // (optional) classifier weights for the logits, on the last layer
    wcls: Range<usize>,
}

impl TransformerWeights {
    fn try_from(data: Vec<f32>, config: &Config) -> Result<Self, std::io::Error> {
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let n_layers = config.n_layers as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let seq_len = config.seq_len as usize;
        let vocab_size = config.vocab_size as usize;
        let head_size = dim / n_heads;

        let mut start: usize = 7; // Config size, 7 i32
        let mut end: usize = start + (vocab_size * dim);

        let shared_start = start;
        let shared_end = end;

        let token_embedding_table = start..end;

        start = end;
        end += n_layers * dim;
        let rms_att_weight = start..end;

        start = end;
        end += n_layers * dim * (n_heads * head_size);
        let wq = start..end;

        start = end;
        end += n_layers * dim * (n_kv_heads * head_size);
        let wk = start..end;

        start = end;
        end += n_layers * dim * (n_kv_heads * head_size);
        let wv = start..end;

        start = end;
        end += n_layers * (n_heads * head_size) * dim;
        let wo = start..end;

        start = end;
        end += n_layers * dim;
        let rms_ffn_weight = start..end;

        start = end;
        end += n_layers * dim * hidden_dim;
        let w1 = start..end;

        start = end;
        end += n_layers * hidden_dim * dim;
        let w2 = start..end;

        start = end;
        end += n_layers * dim * hidden_dim;
        let w3 = start..end;

        start = end;
        end += dim + seq_len * head_size;
        let rms_final_weight = start..end;

        let wcls = if config.shared_weights {
            shared_start..shared_end
        } else {
            end..data.len()
        };

        Ok(Self {
            data,
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }

    fn token_embedding_table(&self) -> &[f32] {
        &self.data[self.token_embedding_table.clone()]
    }

    fn rms_att_weight(&self) -> &[f32] {
        &self.data[self.rms_att_weight.clone()]
    }

    fn rms_ffn_weight(&self) -> &[f32] {
        &self.data[self.rms_ffn_weight.clone()]
    }

    fn wq(&self) -> &[f32] {
        &self.data[self.wq.clone()]
    }

    fn wk(&self) -> &[f32] {
        &self.data[self.wk.clone()]
    }

    fn wv(&self) -> &[f32] {
        &self.data[self.wv.clone()]
    }

    fn wo(&self) -> &[f32] {
        &self.data[self.wo.clone()]
    }

    fn w1(&self) -> &[f32] {
        &self.data[self.w1.clone()]
    }

    fn w2(&self) -> &[f32] {
        &self.data[self.w2.clone()]
    }

    fn w3(&self) -> &[f32] {
        &self.data[self.w3.clone()]
    }

    fn rms_final_weight(&self) -> &[f32] {
        &self.data[self.rms_final_weight.clone()]
    }

    fn wcls(&self) -> &[f32] {
        &self.data[self.wcls.clone()]
    }
}

struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits

    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(
        dim: usize,
        hidden_dim: usize,
        att_size: usize,
        vocab_size: usize,
        kv_size: usize,
    ) -> Self {
        Self {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            att: vec![0.0; att_size],
            logits: vec![0.0; vocab_size],
            key_cache: vec![0.0; kv_size],
            value_cache: vec![0.0; kv_size],
        }
    }
}

struct Transformer {
    config: Config,
    weights: TransformerWeights,
    state: RunState,
}

impl Transformer {
    fn new(config: Config, weights: TransformerWeights, state: RunState) -> Self {
        Self {
            config,
            weights,
            state,
        }
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    assert_eq!(o.len(), x.len());
    assert_eq!(o.len(), weight.len());

    // calculate sum of squares
    let mut ss = 0.0f32;
    for x_v in x {
        ss += x_v * x_v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0f32 / ss.sqrt();

    // normalize and scale
    for ((o_v, w_v), x_v) in o.iter_mut().zip(weight.iter()).zip(x.iter()) {
        *o_v = w_v * (ss * x_v);
    }
}

fn rmsnorm_inplace(x: &mut [f32], weight: &[f32]) {
    assert_eq!(x.len(), weight.len());

    // calculate sum of squares
    let mut ss = 0.0f32;
    for x_v in x.iter() {
        ss += x_v * x_v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0f32 / ss.sqrt();

    // normalize and scale
    for (x_v, w_v) in x.iter_mut().zip(weight.iter()) {
        *x_v = w_v * (ss * *x_v);
    }
}

fn softmax(x: &mut [f32]) {
    // find max value (for numerical stability)
    let max_val: f32 = x.iter().copied().reduce(f32::max).unwrap_or(0.0);

    // exp and sum
    let mut sum = 0.0f32;
    for x_v in x.iter_mut() {
        *x_v = (*x_v - max_val).exp();
        sum += *x_v;
    }

    // normalize
    for x_v in x {
        *x_v /= sum;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    for i in 0..d {
        let mut val = 0.0f32;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn forward(transformer: &mut Transformer, token: usize, pos: usize) -> &mut [f32] {
    // a few convenience variables
    let p = &mut transformer.config;
    let w = &mut transformer.weights;
    let s = &mut transformer.state;

    let x = &mut s.x;
    let dim = p.dim as usize;
    let kv_dim = ((p.dim * p.n_kv_heads) / p.n_heads) as usize;
    let kv_mul = (p.n_heads / p.n_kv_heads) as usize;
    let hidden_dim = p.hidden_dim as usize;
    let head_size = dim / p.n_heads as usize;

    // copy the token embedding into x
    let start = token * dim;
    x.copy_from_slice(&w.token_embedding_table()[start..start + dim]);

    // forward all the layers
    for l in 0..p.n_layers as usize {
        // attention rmsnorm
        let start = l * dim;
        rmsnorm(&mut s.xb, x, &w.rms_att_weight()[start..start + dim]);

        // key and value point to the kv cache
        let loff = l * p.seq_len as usize * kv_dim;
        let start = loff + pos * kv_dim;

        let k = &mut s.key_cache[start..start + kv_dim];
        let v = &mut s.value_cache[start..start + kv_dim];

        // qkv matmuls for this position
        let start = l * dim * dim;
        matmul(&mut s.q, &s.xb, &w.wq()[start..start + dim * dim], dim, dim);

        let start = l * kv_dim * dim;
        matmul(k, &s.xb, &w.wk()[start..start + kv_dim * dim], dim, kv_dim);
        matmul(v, &s.xb, &w.wv()[start..start + kv_dim * dim], dim, kv_dim);

        // RoPE relative positional encoding:: complex-valued rotate q and k in each head
        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0f32 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < kv_dim { 2 } else { 1 };
            for v in 0..rotn {
                let vec = if v == 0 { &mut s.q } else { &mut *k };
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        for h in 0..p.n_heads as usize {
            // get the query vector for this head
            let start = h * head_size;
            let q = &s.q[start..start + head_size];

            // attention scores for this head
            let start = h * p.seq_len as usize;
            let att = &mut s.att[start..start + p.seq_len as usize];

            // iterate over all timesteps, including the current one
            for t in 0..=pos {
                // get the key vector for this head and at this timestep
                let start = loff + t * kv_dim + (h / kv_mul) * head_size;
                let k = &s.key_cache[start..start + head_size];

                // calculate the attention score as the dot product of q and k
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut att[..pos + 1]);

            // weighted sum of the values, store back into xb
            let start = h * head_size;
            let xb = &mut s.xb[start..start + head_size];
            xb.fill(0.0);

            for t in 0..=pos {
                // get the value vector for this head and at this timestep
                let start = loff + t * kv_dim + (h / kv_mul) * head_size;
                let v = &s.value_cache[start..start + head_size];

                // get the attention weight for this timestep
                let a = att[t];
                // accumulate the weighted value into xb
                for i in 0..head_size {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        let start = l * dim * dim;
        matmul(
            &mut s.xb2,
            &s.xb,
            &w.wo()[start..start + dim * dim],
            dim,
            dim,
        );

        // residual connection back into x
        for i in 0..dim {
            x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        let start = l * dim;
        rmsnorm(&mut s.xb, &x, &w.rms_ffn_weight()[start..start + dim]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        let start = l * dim * hidden_dim;
        let end = start + dim * hidden_dim;
        matmul(&mut s.hb, &s.xb, &w.w1()[start..end], dim, hidden_dim);
        matmul(&mut s.hb2, &s.xb, &w.w3()[start..end], dim, hidden_dim);

        // SwiGLU non-linearity
        for i in 0..hidden_dim {
            let mut val = s.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= 1.0 / (1.0 + (-val).exp());
            // elementwise multiply with w3(x)
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        let start = l * dim * hidden_dim;
        let end = start + hidden_dim * dim;
        matmul(&mut s.xb, &s.hb, &w.w2()[start..end], hidden_dim, dim);

        // residual connection
        for i in 0..dim {
            x[i] += s.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm_inplace(x, &w.rms_final_weight()[..dim]);

    // classifier into logits
    matmul(
        &mut s.logits,
        x,
        w.wcls(),
        p.dim as usize,
        p.vocab_size as usize,
    );

    &mut s.logits
}

struct TokenIndex {
    str: Vec<u8>,
    id: i32,
}

struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    vocab_scores: Vec<f32>,
    sorted_vocab: Vec<TokenIndex>,
    vocab_size: usize,
    max_token_length: usize,
    byte_pieces: Vec<u8>,
}

impl Tokenizer {
    fn new() -> Self {
        Tokenizer {
            vocab: vec![],
            vocab_scores: vec![],
            sorted_vocab: vec![],
            vocab_size: 0,
            max_token_length: 0,
            byte_pieces: vec![],
        }
    }
}

fn build_tokenizer(t: &mut Tokenizer, tokenizer_path: &str, vocab_size: usize) {
    t.vocab_size = vocab_size;
    for i in 0..256 {
        t.byte_pieces.push(i as u8);
        t.byte_pieces.push('\0' as u8);
    }

    let mut file = match File::open(tokenizer_path) {
        Ok(file) => file,
        Err(e) => panic!("Error opening file: {}", e),
    };

    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes).unwrap();
    t.max_token_length = i32::from_le_bytes(bytes) as usize;

    for _ in 0..vocab_size {
        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes).unwrap();
        t.vocab_scores.push(f32::from_le_bytes(bytes));

        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes).unwrap();
        let len = i32::from_le_bytes(bytes) as usize;

        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).unwrap();
        t.vocab.push(bytes);
    }
}

fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn decode(t: &Tokenizer, prev_token: i32, token: i32) -> &[u8] {
    let mut piece: &[u8] = &t.vocab[token as usize];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if prev_token == 1 && piece[0] == ' ' as u8 {
        piece = &t.vocab[token as usize][1..];
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    if let [b'<', b'0', b'x', hi, lo, b'>'] = piece {
        if let (Some(hi), Some(lo)) = (hex_digit(*hi), hex_digit(*lo)) {
            let byte = (hi << 4) | lo;
            let start = usize::from(byte) * 2;
            return &t.byte_pieces[start..start + 1];
        }
    }
    piece
}

fn safe_print(piece: &[u8]) -> io::Result<()> {
    if piece.is_empty() {
        return Ok(());
    }

    if let [byte] = piece {
        if !(byte.is_ascii_graphic()
            || *byte == b' '
            || byte.is_ascii_whitespace()
            || *byte == 0x0b)
        {
            return Ok(());
        }
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    out.write_all(piece)
}

fn str_lookup(piece: &[u8], sorted_vocab: &[TokenIndex], vocab_size: usize) -> Option<i32> {
    sorted_vocab
        .binary_search_by(|x| x.str.as_slice().cmp(piece))
        .ok()
        .map(|index| sorted_vocab[index].id)
}

fn encode(
    t: &mut Tokenizer,
    text: &[u8],
    bos: i8,
    eos: i8,
    tokens: &mut [i32],
    n_tokens: &mut i32,
) {
    if t.sorted_vocab.is_empty() {
        for i in 0..t.vocab_size {
            t.sorted_vocab.push(TokenIndex {
                id: i as i32,
                str: t.vocab[i].clone(),
            });
        }

        t.sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
    }

    let mut str_buffer: Vec<u8> = Vec::new();

    // start at 0 tokens
    let mut n_tokens_val = 0;

    // add optional BOS (=1) token, if desired
    if bos != 0 {
        tokens[n_tokens_val] = 1;
        n_tokens_val += 1;
    }

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if text.first().is_some_and(|&b| b != 0) {
        let dummy_prefix = str_lookup(b" ", &t.sorted_vocab, t.vocab_size);
        tokens[n_tokens_val] = dummy_prefix.unwrap();
        n_tokens_val += 1;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (i, &c) in text.iter().enumerate() {
        if c == 0 {
            break;
        }

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if c & 0xc0 != 0x80 {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_buffer.clear();
        }

        // append the current byte to the buffer
        str_buffer.push(c);

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if text.get(i + 1).is_some_and(|&next| (next & 0xC0) == 0x80) && str_buffer.len() < 4 {
            continue;
        }

        let id = str_lookup(&str_buffer, &t.sorted_vocab, t.vocab_size);

        if let Some(id) = id {
            // we found this codepoint in vocab, add it as a token
            tokens[n_tokens_val] = id;
            n_tokens_val += 1;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for &buf in &str_buffer {
                tokens[n_tokens_val] = i32::from(buf) + 3;
                n_tokens_val += 1;
            }
        }
        str_buffer.clear(); // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    loop {
        let mut best_score = -1e10f32;
        let mut best_id = -1;
        let mut best_idx = -1;

        for i in 0..n_tokens_val.saturating_sub(1) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            str_buffer.clear();
            str_buffer.extend_from_slice(&t.vocab[tokens[i] as usize]);
            str_buffer.extend_from_slice(&t.vocab[tokens[i + 1] as usize]);
            let id = str_lookup(&str_buffer, &t.sorted_vocab, t.vocab_size);
            if let Some(id) = id
                && t.vocab_scores[id as usize] > best_score
            {
                // this merge pair exists in vocab! record its score and position
                best_score = t.vocab_scores[id as usize];
                best_id = id;
                best_idx = i as i32;
            }
        }

        if best_idx == -1 {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx as usize] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for i in (best_idx + 1) as usize..n_tokens_val - 1 {
            tokens[i] = tokens[i + 1];
        }
        n_tokens_val -= 1;
    }

    // add optional EOS (=2) token, if desired
    if eos != 0 {
        tokens[n_tokens_val] = 2;
        n_tokens_val += 1;
    }

    *n_tokens = i32::try_from(n_tokens_val).expect("n_token_val exceeds i32");
}

struct ProbIndex {
    prob: f32,
    index: usize,
}

struct Sampler {
    vocab_size: usize,
    probindex: Vec<ProbIndex>, // buffer used in top-p sampling
    temperature: f32,
    topp: f32,
}

impl Sampler {
    fn new(vocab_size: usize, temperature: f32, topp: f32) -> Self {
        Self {
            vocab_size,
            probindex: vec![],
            temperature,
            topp,
        }
    }
}

fn sample_argmax(probabilities: &[f32], n: usize) -> usize {
    // return the index that has the highest probability
    let mut max_i = 0;
    let mut max_p = probabilities[0];
    for i in 1..n {
        if probabilities[i] > max_p {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    max_i
}

fn sample_mult(probabilities: &[f32], n: usize, coin: f32) -> usize {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if coin < cdf {
            return i;
        }
    }
    n - 1 // in case of rounding errors
}

fn sample_topp(
    probabilities: &[f32],
    n: usize,
    topp: f32,
    probindex: &mut Vec<ProbIndex>,
    coin: f32,
) -> usize {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()
    probindex.clear();

    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    let cutoff = (1.0f32 - topp) / (n - 1) as f32;
    for i in 0..n {
        if probabilities[i] >= cutoff {
            probindex.push(ProbIndex {
                index: i,
                prob: probabilities[i],
            });
        }
    }
    probindex.sort_by(|a, b| b.prob.total_cmp(&a.prob));

    // truncate the list where cumulative probability exceeds topp
    let mut cumulative_prob = 0.0f32;
    let mut last_index = probindex.len() - 1; // in case of rounding errors consider all elements
    for i in 0..probindex.len() {
        cumulative_prob += probindex[i].prob;
        if cumulative_prob > topp {
            last_index = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    let r = coin * cumulative_prob;
    let mut cdf = 0.0f32;
    for i in 0..=last_index {
        cdf += probindex[i].prob;
        if r < cdf {
            return probindex[i].index;
        }
    }
    probindex[last_index].index // in case of rounding errors
}

fn sample(sampler: &mut Sampler, logits: &mut [f32]) -> usize {
    // sample the token given the logits and some hyperparameters
    let next;
    if sampler.temperature == 0.0 {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler.vocab_size);
    } else {
        // apply the temperature to the logits
        for q in 0..sampler.vocab_size {
            logits[q] /= sampler.temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(&mut logits[..sampler.vocab_size]);
        // flip a (float) coin (this is our source of entropy for sampling)
        let coin: f32 = rand::random();
        // we sample from this distribution to get the next token
        if sampler.topp <= 0.0 || sampler.topp >= 1.0 {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler.vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(
                logits,
                sampler.vocab_size,
                sampler.topp,
                &mut sampler.probindex,
                coin,
            );
        }
    }
    next
}

fn generate(
    mut transformer: Transformer,
    mut tokenizer: Tokenizer,
    mut sampler: Sampler,
    prompt: &[u8],
    steps: i32,
) {
    let mut num_prompt_tokens = 0;
    let mut prompt_tokens: Vec<i32> = vec![0; prompt.len() + 3];
    encode(
        &mut tokenizer,
        prompt,
        1,
        0,
        &mut prompt_tokens,
        &mut num_prompt_tokens,
    );
    if num_prompt_tokens < 1 {
        panic!("something is wrong, expected at least 1 prompt token")
    }

    // start the main loop
    let mut next = 0;
    let mut token = prompt_tokens[0];
    let mut pos = 0usize;
    while pos < steps as usize {
        // forward the transformer to get logits for the next token
        let logits: &mut [f32] = forward(&mut transformer, token as usize, pos);

        // advance the state machine
        if pos < (num_prompt_tokens - 1) as usize {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(&mut sampler, logits.as_mut()) as i32;
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        let piece = decode(&tokenizer, token, next);
        safe_print(piece).expect("Failed to flush stdout!");
        token = next;
    }
    println!()
}

fn main() {
    let path = Path::new("stories15M.bin");
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => panic!("Error opening file: {}", e),
    };

    let mut bytes = [0u8; 28];
    file.read_exact(&mut bytes).unwrap();

    let config = Config::try_from(&bytes).unwrap();
    let vocab_size = config.vocab_size as usize;

    let data: Vec<u8> = fs::read(path).unwrap();
    let data: Vec<f32> = data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let weights = TransformerWeights::try_from(data, &config).unwrap();

    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    let state = RunState::new(
        config.dim as usize,
        config.hidden_dim as usize,
        (config.n_heads * config.seq_len) as usize,
        vocab_size,
        (config.n_layers * config.seq_len * kv_dim) as usize,
    );

    let transformer = Transformer::new(config, weights, state);

    // tokenizer
    let tokenizer_path = "tokenizer.bin";
    let mut tokenizer = Tokenizer::new();
    build_tokenizer(&mut tokenizer, tokenizer_path, vocab_size);

    // sampler
    let sampler = Sampler::new(vocab_size, 1.0, 0.9);

    // run!
    let prompt = String::from("Hello World! ");
    generate(transformer, tokenizer, sampler, prompt.as_bytes(), 256);
}
