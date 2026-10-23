use std::path::Path;

use crate::{Config, Transformer, TransformerWeights};

use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::{reduce_sum, *};

    #[cutile::entry()]
    fn softmax<const BM: i32>(y: &mut Tensor<f32, { [BM] }>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = x.load_like(y);

        let tile_x_max: Tile<f32, { [1] }> = reduce_max(tile_x, 0);
        let tile_x_max = tile_x_max.broadcast(y.shape());

        let num = exp(tile_x - tile_x_max);

        let denom: Tile<f32, { [1] }> = reduce_sum(num, 0);
        let denom = denom.broadcast(y.shape());

        y.store(num / denom);
    }

    #[cutile::entry()]
    fn rmsnorm<const BM: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1] }>,
        w: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(y);
        let tile_w = w.load_like(y);

        let len = constant(x.shape()[0] as f32, shape![1]);
        let ep = constant(1e-5 as f32, shape![1]);
        let one = constant(1.0f32, shape![1]);

        let tile_x_sum: Tile<f32, { [1] }> = reduce_sum(tile_x * tile_x, 0);
        let tile_x_sum = tile_x_sum / len + ep;

        let ss = one / sqrt(tile_x_sum, rounding::Zero, ftz::Disabled);
        let ss = ss.broadcast(shape![BM]);

        y.store(tile_w * ss * tile_x);
    }

    #[cutile::entry()]
    fn matmul_1d<const BM: i32, const DIM: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        w: &Tensor<f32, { [-1, -1, -1] }>,
        x: &Tensor<f32, { [-1] }>,
        l: i32,
    ) {
        let w_part = w.partition(shape![1, BM, DIM]);
        let tw = w_part.load([l, 0, 0]);
        let tw = tw.reshape(shape![BM, DIM]);

        let x_part = x.partition(shape![DIM]);
        let tx = x_part.load([0]);
        let tx = tx.reshape(shape![DIM, 1]);

        let out = constant(0.0f32, shape![BM, 1]);
        let out = mma(tw, tx, out);
        let out = out.reshape(shape![BM]);

        y.store(out);
    }

    #[cutile::entry()]
    fn matmul<const SEQ_LEN: i32, const BM: i32, const DIM: i32>(
        y: &mut Tensor<f32, { [1, 1, BM] }>,
        w: &Tensor<f32, { [-1, -1, -1] }>,
        x: &Tensor<f32, { [-1] }>,
        l: i32,
        pos: i32,
    ) {
        if l != program_id(0) || pos != program_id(1) {
            return;
        }

        let w_part = w.partition(shape![1, BM, DIM]);
        let tw = w_part.load([l, 0, 0]);
        let tw = tw.reshape(shape![BM, DIM]);

        let x_part = x.partition(shape![DIM]);
        let tx = x_part.load([0]);
        let tx = tx.reshape(shape![DIM, 1]);

        let out = constant(0.0f32, shape![BM, 1]);
        let out = mma(tw, tx, out);
        let out = out.reshape(shape![1, 1, BM]);

        y.store(out);
    }

    #[cutile::entry()]
    fn rope_q<const HEAD_SIZE: i32, const HALF_DIM: i32>(
        q: &mut Tensor<f32, { [HEAD_SIZE] }>,
        pos: i32,
    ) {
        let head_dim: Tile<f32, { [HALF_DIM] }> = iota(shape![HALF_DIM]);
        let head_dim = head_dim * constant(2.0f32, shape![HALF_DIM]); // step by 2
        let head_size = constant(HEAD_SIZE as f32, shape![HALF_DIM]);
        let exponent = head_dim / head_size;

        let base = constant(10000.0f32, shape![HALF_DIM]);
        let freq = constant(1.0f32, shape![HALF_DIM]) / pow(base, exponent);

        let val = constant(pos as f32, shape![HALF_DIM]) * freq;
        let fcr = cos(val).reshape(shape![HALF_DIM, 1]);
        let fci = sin(val).reshape(shape![HALF_DIM, 1]);

        // q
        let pairs = q.load();
        let pairs = pairs.reshape(shape![HALF_DIM, 2]);

        let zero = constant(0i32, shape![]);
        let one = constant(1i32, shape![]);

        let v0: Tile<f32, { [HALF_DIM, 1] }> = extract(pairs, [zero, zero]);
        let v1: Tile<f32, { [HALF_DIM, 1] }> = extract(pairs, [zero, one]);

        let r0 = v0 * fcr - v1 * fci;
        let r1 = v0 * fci + v1 * fcr;

        let rotated: Tile<f32, { [HALF_DIM, 2] }> = cat(r0, r1, 1);
        q.store(rotated.reshape(shape![HEAD_SIZE]));
    }

    #[cutile::entry()]
    fn rope_k<const HEAD_SIZE: i32, const HALF_DIM: i32>(
        k: &mut Tensor<f32, { [1, 1, HEAD_SIZE] }>,
        l: i32,
        pos: i32,
    ) {
        if l != program_id(0) || pos != program_id(1) {
            return;
        }

        let head_dim: Tile<f32, { [HALF_DIM] }> = iota(shape![HALF_DIM]);
        let head_dim = head_dim * constant(2.0f32, shape![HALF_DIM]); // step by 2
        let head_size = constant(HEAD_SIZE as f32, shape![HALF_DIM]);
        let exponent = head_dim / head_size;

        let base = constant(10000.0f32, shape![HALF_DIM]);
        let freq = constant(1.0f32, shape![HALF_DIM]) / pow(base, exponent);

        let val = constant(pos as f32, shape![HALF_DIM]) * freq;
        let fcr = cos(val).reshape(shape![HALF_DIM, 1]);
        let fci = sin(val).reshape(shape![HALF_DIM, 1]);

        let pairs = k.load();
        let pairs = pairs.reshape(shape![HEAD_SIZE]);
        let pairs = pairs.reshape(shape![HALF_DIM, 2]);

        let zero = constant(0i32, shape![]);
        let one = constant(1i32, shape![]);

        let v0: Tile<f32, { [HALF_DIM, 1] }> = extract(pairs, [zero, zero]);
        let v1: Tile<f32, { [HALF_DIM, 1] }> = extract(pairs, [zero, one]);

        let r0 = v0 * fcr - v1 * fci;
        let r1 = v0 * fci + v1 * fcr;

        let rotated: Tile<f32, { [HALF_DIM, 2] }> = cat(r0, r1, 1);
        k.store(rotated.reshape(shape![1, 1, HEAD_SIZE]));
    }

    #[cutile::entry()]
    fn mha<const HEAD_SIZE: i32, const BM: i32>(
        out: &mut Tensor<f32, { [1, HEAD_SIZE] }>,
        q: &Tensor<f32, { [-1, -1] }>,         // n_heads, head_size
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // n_layer, seq_len, n_kv_heads, head_size
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // n_layer, seq_len, n_kv_heads, head_size
        l: i32,
        pos: i32,
        kv_mul: i32,
        qk_scale: f32,
    ) {
        let pid0 = program_id(0);

        let two = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: f32 = qk_scale / log2;
        let qk_scale = qk_scale.broadcast(shape![BM]);

        let tq = q.partition(shape![1, HEAD_SIZE]);
        let tq = tq.load([pid0, 0]);

        let k_part = k.partition_permuted(shape![1, 1, BM, HEAD_SIZE], const_array![0, 2, 1, 3]);
        let v_part = v.partition_permuted(shape![1, 1, BM, HEAD_SIZE], const_array![0, 2, 1, 3]);

        let mut m_i = constant(f32::NEG_INFINITY, shape![1]); // running max
        let mut l_i = constant(0.0f32, shape![1]); // rescale factor
        let mut acc = constant(0.0f32, shape![1, HEAD_SIZE]); // accumulation

        let num_tiles = ceil_div(pos + 1, BM);

        for j in 0..num_tiles {
            let kv_h_i = pid0 / kv_mul;
            let tk = k_part
                .load([l, kv_h_i, j, 0])
                .reshape(shape![BM, HEAD_SIZE]);
            let tk_transpose = tk.transpose();
            let qk = constant(0.0f32, shape![1, BM]);
            let qk = mma(tq, tk_transpose, qk).reshape(shape![BM]);
            let qk = qk * qk_scale;

            // casual masking
            let offsets = iota(shape![BM]);
            let token_idx = offsets + broadcast_scalar(j * BM, shape![BM]);

            let valid = lt_tile(token_idx, broadcast_scalar(pos + 1, shape![BM]));

            let neg_inf = constant(f32::NEG_INFINITY, shape![BM]);
            let qk = select(valid, qk, neg_inf);

            // update running max
            let qk_max: Tile<f32, { [1] }> = reduce_max(qk, 0);
            let m_ij = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM]);

            // Softmax numerator and correction factor
            let p = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [1] }> = reduce_sum(p, 0);
            let alpha = exp2(m_i - m_ij, ftz::Disabled);

            // update running sum and rescale accumulator
            l_i = l_i * alpha + l_ij;
            let alpha = alpha
                .broadcast(shape![HEAD_SIZE])
                .reshape(shape![1, HEAD_SIZE]);
            acc = acc * alpha;

            // Accumulate P @ V
            let tv = v_part
                .load([l, kv_h_i, j, 0])
                .reshape(shape![BM, HEAD_SIZE]);
            let p = p.reshape(shape![1, BM]);
            acc = mma(p, tv, acc);
            m_i = m_ij;
        }

        // Final normalization
        acc = true_div(
            acc,
            l_i.broadcast(shape![HEAD_SIZE])
                .reshape(shape![1, HEAD_SIZE]),
        );
        out.store(acc);
    }

    #[cutile::entry()]
    fn fmatmul<const DIM: i32>(
        out: &mut Tensor<f32, { [DIM] }>,
        x: &Tensor<f32, { [-1] }>,          // dim
        wo: &Tensor<f32, { [-1, -1, -1] }>, // n_layer, dim, dim
        l: i32,
    ) {
        let tx = x.load_like(out).reshape(shape![1, DIM]);

        let tw = wo.partition(shape![1, DIM, DIM]);
        let tw = tw.load([l, 0, 0]).reshape(shape![DIM, DIM]);

        let mut y = constant(0.0f32, shape![1, DIM]);
        y = mma(tx, tw, y);

        let residual = out.load();
        out.store(residual + y.reshape(shape![DIM]));
    }

    #[cutile::entry()]
    fn fffn<const DIM: i32, const HIDDEN_DIM: i32>(
        out: &mut Tensor<f32, { [DIM] }>,
        x: &Tensor<f32, { [-1] }>,          // dim
        w1: &Tensor<f32, { [-1, -1, -1] }>, // layer, hidden_dim, dim
        w2: &Tensor<f32, { [-1, -1, -1] }>, // layer, dim, hidden_dim
        w3: &Tensor<f32, { [-1, -1, -1] }>, // layer, hidden_dim, dim
        l: i32,
    ) {
        let tx = x.load_like(out).reshape(shape![DIM, 1]);

        let tw1 = w1.partition(shape![1, HIDDEN_DIM, DIM]);
        let tw1 = tw1.load([l, 0, 0]).reshape(shape![HIDDEN_DIM, DIM]);

        let tw2 = w2.partition(shape![1, DIM, HIDDEN_DIM]);
        let tw2 = tw2.load([l, 0, 0]).reshape(shape![DIM, HIDDEN_DIM]);

        let tw3 = w3.partition(shape![1, HIDDEN_DIM, DIM]);
        let tw3 = tw3.load([l, 0, 0]).reshape(shape![HIDDEN_DIM, DIM]);

        // matmul
        let w1x = constant(0.0f32, shape![HIDDEN_DIM, 1]);
        let w1x = mma(tw1, tx, w1x);

        let w3x = constant(0.0f32, shape![HIDDEN_DIM, 1]);
        let w3x = mma(tw3, tx, w3x);

        // SwiGLU non-linearity
        let one = constant(1.0f32, w1x.shape());
        let w1x = (w1x / (one + exp(negf(w1x)))) * w3x;

        let y = constant(0.0f32, shape![DIM, 1]);
        let y = mma(tw2, w1x, y);
        let y = y.reshape(shape![DIM]);

        let input = out.load();
        out.store(input + y);
    }
}

struct GpuTransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)

    // weights for rmsnorms
    rms_att_weight: Arc<Tensor<f32>>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Arc<Tensor<f32>>, // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size
    wq: Arc<Tensor<f32>>, // (layer, dim, n_heads * head_size)
    wk: Arc<Tensor<f32>>, // (layer, dim, n_kv_heads * head_size)
    wv: Arc<Tensor<f32>>, // (layer, dim, n_kv_heads * head_size)
    wo: Arc<Tensor<f32>>, // (layer, n_heads * head_size, dim)

    // weights for ffn
    w1: Arc<Tensor<f32>>, // (layer, hidden_dim, dim)
    w2: Arc<Tensor<f32>>, // (layer, dim, hidden_dim)
    w3: Arc<Tensor<f32>>, // (layer, hidden_dim, dim)

    // final rmsnorm
    rms_final_weight: Arc<Tensor<f32>>, // (dim,)

    // (optional) classifier weights for the logits, on the last layer
    wcls: Arc<Tensor<f32>>,
}

impl GpuTransformerWeights {
    fn from_weights(weights_host: &TransformerWeights, c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;

        let rms_att_weight =
            api::copy_host_vec_to_device(&Arc::new(weights_host.rms_att_weight().to_vec()))
                .reshape(&[c.n_layers, c.dim])
                .sync()
                .unwrap()
                .into();
        let rms_ffn_weight =
            api::copy_host_vec_to_device(&Arc::new(weights_host.rms_ffn_weight().to_vec()))
                .reshape(&[c.n_layers, c.dim])
                .sync()
                .unwrap()
                .into();

        let wq = api::copy_host_vec_to_device(&Arc::new(weights_host.wq().to_vec()))
            .reshape(&[c.n_layers, c.dim, c.dim])
            .sync()
            .unwrap()
            .into();
        let wk = api::copy_host_vec_to_device(&Arc::new(weights_host.wk().to_vec()))
            .reshape(&[c.n_layers, c.n_kv_heads * head_size, c.dim])
            .sync()
            .unwrap()
            .into();
        let wv = api::copy_host_vec_to_device(&Arc::new(weights_host.wv().to_vec()))
            .reshape(&[c.n_layers, c.n_kv_heads * head_size, c.dim])
            .sync()
            .unwrap()
            .into();
        let wo = api::copy_host_vec_to_device(&Arc::new(weights_host.wo().to_vec()))
            .reshape(&[c.n_layers, c.dim, c.dim])
            .sync()
            .unwrap()
            .into();

        let w1 = api::copy_host_vec_to_device(&Arc::new(weights_host.w1().to_vec()))
            .reshape(&[c.n_layers, c.hidden_dim, c.dim])
            .sync()
            .unwrap()
            .into();
        let w2 = api::copy_host_vec_to_device(&Arc::new(weights_host.w2().to_vec()))
            .reshape(&[c.n_layers, c.dim, c.hidden_dim])
            .sync()
            .unwrap()
            .into();
        let w3 = api::copy_host_vec_to_device(&Arc::new(weights_host.w3().to_vec()))
            .reshape(&[c.n_layers, c.hidden_dim, c.dim])
            .sync()
            .unwrap()
            .into();

        let rms_final_weight =
            api::copy_host_vec_to_device(&Arc::new(weights_host.rms_final_weight().to_vec()))
                .reshape(&[c.dim])
                .sync()
                .unwrap()
                .into();

        let wcls = api::copy_host_vec_to_device(&Arc::new(weights_host.wcls().to_vec()))
            .sync()
            .unwrap()
            .into();

        Self {
            token_embedding_table: weights_host.token_embedding_table().to_vec(),
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
        }
    }
}

struct GpuRunState {
    // kv cache
    key_cache: Arc<Tensor<f32>>,   // (layer, seq_len, dim)
    value_cache: Arc<Tensor<f32>>, // (layer, seq_len, dim)
}

impl GpuRunState {
    fn new(c: &Config) -> Self {
        let kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;

        let key_cache = api::zeros(&[c.n_layers, c.seq_len, kv_dim])
            .sync()
            .unwrap()
            .into();
        let value_cache = api::zeros(&[c.n_layers, c.seq_len, kv_dim])
            .sync()
            .unwrap()
            .into();

        Self {
            key_cache,
            value_cache,
        }
    }
}

pub struct GpuTransformer {
    config: Config,
    weights: GpuTransformerWeights,
    state: GpuRunState,
}

impl GpuTransformer {
    pub fn from_file(model_path: impl AsRef<Path>) -> Self {
        let transformer = Transformer::from_file(model_path);
        let weights =
            GpuTransformerWeights::from_weights(&transformer.weights, &transformer.config);
        let state = GpuRunState::new(&transformer.config);

        Self {
            config: transformer.config,
            weights,
            state,
        }
    }

    pub fn forward(&mut self, token: usize, pos: usize) {
        // a few convenience variables
        let p = &mut self.config;
        let w = &mut self.weights;
        let s = &mut self.state;

        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let head_size = dim / p.n_heads;

        let start = token * dim;
        let x_h = w.token_embedding_table[start..start + dim].to_owned();

        let mut x_d: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_h.clone()))
            .sync()
            .unwrap()
            .into();

        // forward all the layers
        for l in 0..p.n_layers {
            // attention rmsnorm
            let start = l * dim;
            let xb_d = api::zeros(&[dim]).sync().unwrap().partition([dim]);

            let (y, _, _) = kernel::rmsnorm(xb_d, &x_d, &w.rms_att_weight)
                .sync()
                .unwrap();
            let xb_d = y.unpartition();

            // key and value point to the kv cache
            let loff = l * p.seq_len * kv_dim;
            let start = loff + pos * kv_dim;

            // qkv matmuls for this position
            let q_d = api::zeros(&[dim]).sync().unwrap().partition([dim]);
            let (tq, _, _, _) = kernel::matmul_1d(q_d, &w.wq, &xb_d, l as i32)
                .sync()
                .unwrap();

            let cache = Arc::get_mut(&mut s.key_cache).expect("key_cache ownership");
            let tk = cache.partition([1, 1, kv_dim]);
            let (tk, _, _, _, _) = kernel::matmul(tk, &w.wk, &xb_d, l as i32, pos as i32)
                .sync()
                .unwrap();

            let cache = Arc::get_mut(&mut s.value_cache).expect("value_cache ownership");
            let tv = cache.partition([1, 1, kv_dim]);
            let (tv, _, _, _, _) = kernel::matmul(tv, &w.wv, &xb_d, l as i32, pos as i32)
                .sync()
                .unwrap();

            let tq = tq.unpartition().partition([head_size]);
            let tk = tk.unpartition().partition([1, 1, head_size]);

            let generics = vec![head_size.to_string(), (head_size / 2).to_string()];
            let (tq, _) = kernel::rope_q(tq, pos as i32)
                .generics(generics)
                .sync()
                .unwrap();

            let generics = vec![head_size.to_string(), (head_size / 2).to_string()];
            let (tk, _, _) = kernel::rope_k(tk, l as i32, pos as i32)
                .generics(generics)
                .sync()
                .unwrap();

            // multi-head attention
            let tq = tq.unpartition(); // [dim] -> [n_heads * head_size] -> [n_heads, head_size]
            let tk = tk.unpartition(); // [layer, seq_len, kv_dim] -> [layer, seq_len, n_kv_heads * head_size]
            // -> [layer, seq_len, n_kv_heads, head_size] -> [layer, n_kv_heads, seq_len, head_size]
            let tv = tv.unpartition(); // [layer, seq_len, kv_dim] -> [layer, seq_len, n_kv_heads * head_size]
            // -> [layer, seq_len, n_kv_heads, head_size] -> [layer, n_kv_heads, seq_len, head_size]

            // score: tq * tk -> [n_heads, seq_len]
            // score * v -> [n_heads, head_size]

            // out: [n_heads, head_size] -> [n_heads * head_size] -> [dim]
            let att_out = api::zeros(&[p.n_heads, head_size])
                .sync()
                .unwrap()
                .partition([1, head_size]);
            let tq = tq.reshape(&[p.n_heads, head_size]).unwrap();
            let tk = tk
                .view(&[p.n_layers, p.seq_len, p.n_kv_heads, head_size])
                .unwrap();
            let tv = tv
                .view(&[p.n_layers, p.seq_len, p.n_kv_heads, head_size])
                .unwrap();

            let qk_scale = 1.0 / f32::sqrt(head_size as f32);
            let generics = vec![head_size.to_string(), (4).to_string()];
            let (att_out, tq, tk, tv, _, _, _, _) = kernel::mha(
                att_out,
                tq,
                &tk,
                &tv,
                l as i32,
                pos as i32,
                kv_mul as i32,
                qk_scale,
            )
            .generics(generics)
            .sync()
            .unwrap();

            // final matmul to get the output of the attention
            let att_out = att_out.unpartition();
            let x_mut = Arc::get_mut(&mut x_d).expect("x_d must have exclusive ownership");
            let out = x_mut.partition([dim]);

            let (out, _, _, _) = kernel::fmatmul(out, &att_out, &w.wo, l as i32)
                .sync()
                .unwrap();
            let out = out.unpartition();

            // ffn rmsnorm
            let frms_out = api::zeros(&[dim]).partition([dim]).sync().unwrap();
            let (frms_out, _, _) = kernel::rmsnorm(frms_out, &*out, &w.rms_ffn_weight)
                .sync()
                .unwrap();
            let frms_out = frms_out.unpartition();

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            let (_, _, _, _, _, _) = kernel::fffn(
                out.partition([dim]),
                &frms_out,
                &w.w1,
                &w.w2,
                &w.w3,
                l as i32,
            )
            .generics(vec![dim.to_string(), hidden_dim.to_string()])
            .sync()
            .unwrap();
        }
    }
}
