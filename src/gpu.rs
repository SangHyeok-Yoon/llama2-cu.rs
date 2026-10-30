use std::path::Path;

use crate::{Config, Inference, Transformer, TransformerWeights};

use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::{reduce_sum, *};

    #[cutile::entry()]
    fn rmsnorm<const BM: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1] }>,     // dim
        w: &Tensor<f32, { [-1, -1] }>, // layer, dim
        l: i32,
    ) {
        let dim = x.shape()[0];
        let num_tile = ceil_div(dim, BM);

        let x_part = x.partition(shape![BM]);

        let mut x_sum: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        for t in 0..num_tile {
            let tx = x_part.load([t]);
            let tx_sum: Tile<f32, { [1] }> = reduce_sum(tx * tx, 0);
            x_sum = x_sum + tx_sum.reshape(shape![1]);
        }

        let len = broadcast_scalar(convert_scalar::<f32>(dim), shape![1]);
        let eps = constant(1e-5f32, shape![1]);
        let one = constant(1.0f32, shape![1]);

        x_sum = x_sum / len + eps;

        let ss = one / sqrt(x_sum, rounding::Zero, ftz::Disabled);
        let ss = ss.broadcast(shape![BM]);

        let tx = x.load_like(y);
        let w_part = w.partition(shape![1, BM]);
        let tw = w_part.load([l, program_id(0)]).reshape(shape![BM]);

        y.store(tw * ss * tx);
    }

    #[cutile::entry()]
    fn matmul_qk<const BM: i32>(
        y: &mut Tensor<f32, { [BM] }>,     // dim | kv_dim
        w: &Tensor<f32, { [-1, -1, -1] }>, // l, dim | kv_dim, dim
        x: &Tensor<f32, { [-1] }>,         // dim
        l: i32,
    ) {
        let idx = program_id(0);

        let w_part = w.partition(shape![1, BM, BM]);
        let x_part = x.partition(shape![BM]);

        let mut ty = constant(0.0f32, shape![BM, 1]);

        for t in 0..num_tiles(&w_part, 2) {
            let tw = w_part.load([l, idx, t]).reshape(shape![BM, BM]);
            let tx = x_part.load([t]).reshape(shape![BM, 1]);
            ty = mma(tw, tx, ty);
        }

        y.store(ty.reshape(shape![BM]));
    }

    #[cutile::entry()]
    fn matmul_v<const BM: i32, const MAP_SHAPE: [i32; 3]>(
        // l, seq_len, kv_dim
        mut y: MappedPartitionMut<f32, { [1, 1, BM] }, MAP_SHAPE>,
        w: &Tensor<f32, { [-1, -1, -1] }>, // l, kv_dim, dim
        x: &Tensor<f32, { [-1] }>,         // dim
        l: i32,
        pos: i32,
    ) {
        let w_part = w.partition(shape![1, BM, BM]);
        let x_part = x.partition(shape![BM]);

        for idx in y.iter_indices_within([(l, 1i32), (pos, 1i32), (0i32, -1i32)]) {
            let (layer, _seq, kv_tile) = idx.components();

            let mut ty = constant(0.0f32, shape![BM, 1]);

            for t in 0..num_tiles(&w_part, 2) {
                let tw = w_part.load([layer, kv_tile, t]).reshape(shape![BM, BM]);
                let tx = x_part.load([t]).reshape(shape![BM, 1]);

                ty = mma(tw, tx, ty);
            }

            y.store(ty.reshape(shape![1, 1, BM]), idx);
        }
    }

    #[cutile::entry()]
    fn rope_q<const HEAD_TILE: i32, const HEAD_SIZE: i32, const HALF_DIM: i32>(
        q: &mut Tensor<f32, { [1, HEAD_TILE] }>,
        pos: i32,
    ) {
        let head_dim: Tile<i32, { [HALF_DIM] }> = iota(shape![HALF_DIM]);
        let head_dim: Tile<f32, { [HALF_DIM] }> = convert_tile(head_dim);
        let head_dim = head_dim * constant(2.0f32, shape![HALF_DIM]); // step by 2
        let head_size = broadcast_scalar(convert_scalar::<f32>(HEAD_SIZE), shape![HALF_DIM]);
        let exponent = head_dim / head_size;

        let base = constant(10000.0f32, shape![HALF_DIM]);
        let freq = constant(1.0f32, shape![HALF_DIM]) / pow(base, exponent);

        let val = broadcast_scalar(convert_scalar::<f32>(pos), shape![HALF_DIM]) * freq;
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
        q.store(rotated.reshape(shape![1, HEAD_TILE]));
    }

    #[cutile::entry()]
    fn rope_k<
        const HEAD_TILE: i32,
        const HEAD_SIZE: i32,
        const HALF_DIM: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        // n_layers * seq_len, n_kv_heads, head_size
        mut k: MappedPartitionMut<f32, { [1, 1, HEAD_TILE] }, MAP_SHAPE>,
        k_raw: &Tensor<f32, { [-1, -1] }>, // n_kv_heads, head_size
        row: i32,
        pos: i32,
    ) {
        let head_dim: Tile<i32, { [HALF_DIM] }> = iota(shape![HALF_DIM]);
        let head_dim: Tile<f32, { [HALF_DIM] }> = convert_tile(head_dim);
        let head_dim = head_dim * constant(2.0f32, shape![HALF_DIM]); // step by 2
        let head_size = broadcast_scalar(convert_scalar::<f32>(HEAD_SIZE), shape![HALF_DIM]);
        let exponent = head_dim / head_size;

        let base = constant(10000.0f32, shape![HALF_DIM]);
        let freq = constant(1.0f32, shape![HALF_DIM]) / pow(base, exponent);

        let val = broadcast_scalar(convert_scalar::<f32>(pos), shape![HALF_DIM]) * freq;
        let fcr = cos(val).reshape(shape![HALF_DIM, 1]);
        let fci = sin(val).reshape(shape![HALF_DIM, 1]);

        let raw_part = k_raw.partition(shape![1, HEAD_TILE]);

        for idx in k.iter_indices_within([(row, 1i32), (0i32, -1i32), (0i32, 1i32)]) {
            let (_row, head, _head_tile) = idx.components();

            let pairs = raw_part.load([head, 0]).reshape(shape![HALF_DIM, 2]);

            let zero = constant(0i32, shape![]);
            let one = constant(1i32, shape![]);

            let v0: Tile<f32, { [HALF_DIM, 1] }> = extract(pairs, [zero, zero]);
            let v1: Tile<f32, { [HALF_DIM, 1] }> = extract(pairs, [zero, one]);

            let r0 = v0 * fcr - v1 * fci;
            let r1 = v0 * fci + v1 * fcr;

            let rotated: Tile<f32, { [HALF_DIM, 2] }> = cat(r0, r1, 1);

            k.store(rotated.reshape(shape![1, 1, HEAD_TILE]), idx);
        }
    }

    #[cutile::entry()]
    fn mha<const HEAD_TILE: i32, const BM: i32>(
        out: &mut Tensor<f32, { [1, HEAD_TILE] }>, // n_heads, head_size
        q: &Tensor<f32, { [-1, -1] }>,             // n_heads, head_size
        k: &Tensor<f32, { [-1, -1, -1, -1] }>,     // n_layer, seq_len, n_kv_heads, head_size
        v: &Tensor<f32, { [-1, -1, -1, -1] }>,     // n_layer, seq_len, n_kv_heads, head_size
        l: i32,
        pos: i32,
        kv_mul: i32,
        qk_scale: f32,
    ) {
        let h_idx = program_id(0);

        let two = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: f32 = qk_scale / log2;
        let qk_scale = qk_scale.broadcast(shape![BM]);

        let tq = q.partition(shape![1, HEAD_TILE]);
        let tq = tq.load([h_idx, 0]);

        let k_part = k.partition_permuted(shape![1, 1, BM, HEAD_TILE], const_array![0, 2, 1, 3]);
        let v_part = v.partition_permuted(shape![1, 1, BM, HEAD_TILE], const_array![0, 2, 1, 3]);

        let mut m_i = constant(f32::NEG_INFINITY, shape![1]); // running max
        let mut l_i = constant(0.0f32, shape![1]); // rescale factor
        let mut acc = constant(0.0f32, shape![1, HEAD_TILE]); // accumulation

        let num_tiles = ceil_div(pos + 1, BM);

        for j in 0..num_tiles {
            let kv_h_i = h_idx / kv_mul;
            let tk = k_part
                .load([l, kv_h_i, j, 0])
                .reshape(shape![BM, HEAD_TILE]);
            let tk_transpose = tk.transpose();
            let qk = constant(0.0f32, shape![1, BM]);
            let qk = mma(tq, tk_transpose, qk).reshape(shape![BM]);
            let qk = qk * qk_scale;

            // casual masking
            let offsets: Tile<i32, { [BM] }> = iota(shape![BM]);
            let token_idx = offsets + broadcast_scalar(j * BM, shape![BM]);

            let valid = lt_tile(token_idx, broadcast_scalar(pos + 1, shape![BM]));

            let neg_inf: Tile<f32, { [BM] }> = constant(f32::NEG_INFINITY, shape![BM]);
            let qk = select(valid, qk, neg_inf);

            // update running max
            let qk_max: Tile<f32, { [1] }> = reduce_max(qk, 0);
            let qk_max = qk_max.reshape(shape![1]);
            let m_ij = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM]);

            // Softmax numerator and correction factor
            let p = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [1] }> = reduce_sum(p, 0);
            let l_ij = l_ij.reshape(shape![1]);
            let alpha = exp2(m_i - m_ij, ftz::Disabled);

            // update running sum and rescale accumulator
            l_i = l_i * alpha + l_ij;
            let alpha = alpha
                .broadcast(shape![HEAD_TILE])
                .reshape(shape![1, HEAD_TILE]);
            acc = acc * alpha;

            // Accumulate P @ V
            let tv = v_part
                .load([l, kv_h_i, j, 0])
                .reshape(shape![BM, HEAD_TILE]);
            let p = p.reshape(shape![1, BM]);
            acc = mma(p, tv, acc);
            m_i = m_ij;
        }

        // Final normalization
        acc = true_div(
            acc,
            l_i.broadcast(shape![HEAD_TILE])
                .reshape(shape![1, HEAD_TILE]),
        );
        out.store(acc);
    }

    #[cutile::entry()]
    fn fmatmul<const BM: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        w: &Tensor<f32, { [-1, -1, -1] }>, // n_layer, dim, dim
        x: &Tensor<f32, { [-1] }>,         // dim
        l: i32,
    ) {
        let pid0 = program_id(0);
        let dim = w.shape()[1];
        let num_tile = ceil_div(dim, BM);

        let w_part = w.partition(shape![1, BM, BM]);
        let x_part = x.partition(shape![BM]);

        let mut acc = constant(0.0f32, shape![BM, 1]);

        for t in 0..num_tile {
            let tw = w_part.load([l, pid0, t]).reshape(shape![BM, BM]);
            let tx = x_part.load([t]).reshape(shape![BM, 1]);
            acc = mma(tw, tx, acc);
        }

        let residual = y.load();
        y.store(residual + acc.reshape(shape![BM]));
    }

    #[cutile::entry()]
    fn matmul_silu<const BM: i32, const BK: i32>(
        y: &mut Tensor<f32, { [BM] }>,      // hidden_dim
        w1: &Tensor<f32, { [-1, -1, -1] }>, // layer, hidden_dim, dim
        w3: &Tensor<f32, { [-1, -1, -1] }>, // layer, hidden_dim, dim
        x: &Tensor<f32, { [-1] }>,          // dim
        l: i32,
    ) {
        let idx = program_id(0);
        let dim = x.shape()[0];

        let w1_part = w1.partition(shape![1, BM, BK]);
        let w3_part = w3.partition(shape![1, BM, BK]);
        let x_part = x.partition(shape![BK]);

        let mut w1x = constant(0.0f32, shape![BM, 1]);
        let mut w3x = constant(0.0f32, shape![BM, 1]);

        let num_tile = ceil_div(dim, BK);

        for t in 0..num_tile {
            let tw1 = w1_part.load([l, idx, t]).reshape(shape![BM, BK]);
            let tw3 = w3_part.load([l, idx, t]).reshape(shape![BM, BK]);
            let tx = x_part.load([t]).reshape(shape![BK, 1]);

            w1x = mma(tw1, tx, w1x);
            w3x = mma(tw3, tx, w3x);
        }

        let one = constant(1.0f32, w1x.shape());
        let w1x = (w1x / (one + exp(negf(w1x)))) * w3x;

        y.store(w1x.reshape(shape![BM]));
    }

    #[cutile::entry()]
    fn matmul_ffn<const BM: i32, const BK: i32>(
        y: &mut Tensor<f32, { [BM] }>,      // dim
        w2: &Tensor<f32, { [-1, -1, -1] }>, // layer, dim, hidden_dim
        x: &Tensor<f32, { [-1] }>,          // hidden_dim
        l: i32,
    ) {
        let idx = program_id(0);
        let hidden_dim = x.shape()[0];

        let w2_part = w2.partition(shape![1, BM, BK]);
        let x_part = x.partition(shape![BK]);

        let mut acc = constant(0.0f32, shape![BM, 1]);

        let num_tile = ceil_div(hidden_dim, BK);

        for t in 0..num_tile {
            let tw2 = w2_part.load([l, idx, t]).reshape(shape![BM, BK]);
            let tx = x_part.load([t]).reshape(shape![BK, 1]);
            acc = mma(tw2, tx, acc);
        }

        let residual = y.load();
        y.store(residual + acc.reshape(shape![BM]));
    }

    #[cutile::entry()]
    fn rmsnorm_inplace<const BM: i32>(
        x: &mut Tensor<f32, { [BM] }>, // dim
        w: &Tensor<f32, { [-1] }>,     // dim
    ) {
        let dim = w.shape()[0];

        let tx = load_tile(x, shape![BM], [program_id(0)]);
        let tw = w.load_like(x);

        let sum: Tile<f32, { [1] }> = reduce_sum(tx * tx, 0);
        let mean = sum.reshape(shape![1]) / broadcast_scalar(convert_scalar::<f32>(dim), shape![1]);

        let ep = constant(1e-5f32, shape![1]);
        let one = constant(1.0f32, shape![1]);

        let scale = one / sqrt(mean + ep, rounding::Zero, ftz::Disabled);
        let scale = scale.broadcast(shape![BM]);

        x.store(tx * tw * scale);
    }

    #[cutile::entry()]
    fn matmul_cls<const BM: i32, const BK: i32>(
        y: &mut Tensor<f32, { [BM] }>, // vocab_size
        w: &Tensor<f32, { [-1, -1] }>, // vocab_size, dim
        x: &Tensor<f32, { [-1] }>,     // dim
    ) {
        let idx = program_id(0);
        let dim = x.shape()[0];

        let w_part = w.partition(shape![BM, BK]);
        let x_part = x.partition(shape![BK]);

        let num_tiles = ceil_div(dim, BK);

        let mut acc = constant(0.0f32, shape![BM, 1]);

        for t in 0..num_tiles {
            let tw = w_part.load([idx, t]);
            let tx = x_part.load([t]).reshape(shape![BK, 1]);
            acc = mma(tw, tx, acc);
        }

        y.store(acc.reshape(shape![BM]));
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
            .reshape(&[c.vocab_size, c.dim])
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
    key_cache: Tensor<f32>,   // layer * seq_len, n_kv_heads, head_size
    value_cache: Tensor<f32>, // layer, seq_len, kv_dim

    // buffers
    x_d: Tensor<f32>,           // dim
    xb_d: Option<Tensor<f32>>,  // dim
    q_d: Option<Tensor<f32>>,   // dim
    key_raw: Tensor<f32>,       // kv_dim
    hb_d: Tensor<f32>,          // hidden_dim
    logits_d: Arc<Tensor<f32>>, // vocab_size
}

impl GpuRunState {
    fn new(c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;
        let kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;

        let key_cache = api::zeros(&[c.n_layers * c.seq_len, c.n_kv_heads, head_size])
            .sync()
            .unwrap();
        let value_cache = api::zeros(&[c.n_layers, c.seq_len, kv_dim]).sync().unwrap();

        let x_d = api::zeros::<f32>(&[c.dim]).sync().unwrap();
        let xb_d = api::zeros::<f32>(&[c.dim]).sync().unwrap();
        let q_d = api::zeros::<f32>(&[c.dim]).sync().unwrap();
        let key_raw = api::zeros(&[kv_dim]).sync().unwrap();
        let hb_d = api::zeros::<f32>(&[c.hidden_dim]).sync().unwrap();
        let logits_d = api::zeros::<f32>(&[c.vocab_size]).sync().unwrap();

        Self {
            key_cache,
            value_cache,
            x_d,
            xb_d: Some(xb_d),
            q_d: Some(q_d),
            key_raw,
            hb_d,
            logits_d: Arc::new(logits_d),
        }
    }
}

pub struct GpuTransformer {
    config: Config,
    weights: GpuTransformerWeights,
    state: GpuRunState,
    logits: Vec<f32>,
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
            logits: vec![],
        }
    }
}

impl Inference for GpuTransformer {
    fn get_vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn forward(&mut self, token: usize, pos: usize) -> &mut [f32] {
        // a few convenience variables
        let p = &mut self.config;
        let w = &mut self.weights;
        let s = &mut self.state;

        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let head_size = dim / p.n_heads;

        let start = token * dim;
        let x_h = w.token_embedding_table[start..start + dim].to_owned();

        // buffers
        let mut key_cache = &mut s.key_cache;
        let mut val_cache = &mut s.value_cache;

        let mut x_d = &mut s.x_d;
        let mut xb_d = s.xb_d.take().unwrap();
        let mut q_d = s.q_d.take().unwrap();
        let mut key_raw = &mut s.key_raw;
        let mut hb_d = &mut s.hb_d;
        let logits_d = &mut s.logits_d;

        // copy input
        *x_d = api::copy_host_vec_to_device(&Arc::new(x_h)).sync().unwrap();

        // tile size
        let dim_ts = dim.next_power_of_two() / 2;
        let kv_dim_ts = kv_dim.next_power_of_two() / 2;
        let head_size_ts = head_size.next_power_of_two();

        // forward all the layers
        for l in 0..p.n_layers {
            // attention rmsnorm
            let xb_p = (&mut xb_d).partition([dim_ts]);
            let _ = kernel::rmsnorm(xb_p, &*x_d, &w.rms_att_weight, l as i32)
                .first()
                .sync()
                .unwrap();

            // qkv matmuls for this position
            let q_p = (&mut q_d).partition([dim_ts]);
            let _ = kernel::matmul_qk(q_p, &w.wq, &xb_d, l as i32)
                .first()
                .sync()
                .unwrap();

            let num_tile_blocks = kv_dim.div_ceil(kv_dim_ts);

            let k_raw_p = (&mut key_raw).partition([kv_dim_ts]);
            let _ = kernel::matmul_qk(k_raw_p, &w.wk, &xb_d, l as i32)
                .first()
                .sync()
                .unwrap();

            let v_p = (&mut val_cache)
                .partition([1, 1, kv_dim_ts])
                .map([1, 1, 1], num_tile_blocks as u32);
            let _ = kernel::matmul_v(v_p, &w.wv, &xb_d, l as i32, pos as i32)
                .first()
                .sync()
                .unwrap();

            // rope q
            q_d = q_d.reshape(&[p.n_heads, head_size]).unwrap();
            let q_p = (&mut q_d).partition([1, head_size_ts]);

            let generics = vec![
                head_size_ts.to_string(),
                head_size.to_string(),
                (head_size_ts / 2).to_string(),
            ];
            let _ = kernel::rope_q(q_p, pos as i32)
                .generics(generics)
                .first()
                .sync()
                .unwrap();

            // rope k
            let map_shape = [1, 1, 1];
            let k_p = (&mut key_cache)
                .partition([1, 1, head_size_ts])
                .map(map_shape, p.n_kv_heads as u32);

            let k_raw_v = key_raw.view(&[p.n_kv_heads, head_size]).unwrap();

            let generics = vec![
                head_size_ts.to_string(),
                head_size.to_string(),
                (head_size_ts / 2).to_string(),
                map_shape[0].to_string(),
                map_shape[1].to_string(),
                map_shape[2].to_string(),
            ];

            let _ = kernel::rope_k(k_p, &k_raw_v, (l * p.seq_len + pos) as i32, pos as i32)
                .generics(generics)
                .first()
                .sync()
                .unwrap();

            // multi-head attention
            // q: [dim] -> [n_heads * head_size] -> [n_heads, head_size]
            // k: [layer, seq_len, kv_dim] -> [layer, seq_len, n_kv_heads * head_size]
            // -> [layer, seq_len, n_kv_heads, head_size] -> [layer, n_kv_heads, seq_len, head_size]
            // v: [layer, seq_len, kv_dim] -> [layer, seq_len, n_kv_heads * head_size]
            // -> [layer, seq_len, n_kv_heads, head_size] -> [layer, n_kv_heads, seq_len, head_size]

            // score: tq * tk -> [n_heads, seq_len]
            // score * v -> [n_heads, head_size]

            // out: [n_heads, head_size] -> [n_heads * head_size] -> [dim]
            let k_v = key_cache
                .view(&[p.n_layers, p.seq_len, p.n_kv_heads, head_size])
                .unwrap();
            let v_v = val_cache
                .view(&[p.n_layers, p.seq_len, p.n_kv_heads, head_size])
                .unwrap();

            xb_d = xb_d.reshape(&[p.n_heads, head_size]).unwrap();
            let xb_part = (&mut xb_d).partition([1, head_size_ts]);

            let qk_scale = 1.0 / f32::sqrt(head_size as f32);
            let generics = vec![head_size_ts.to_string(), (4).to_string()];
            let (_, _, _, _, _, _, _, _) = kernel::mha(
                xb_part,
                &q_d,
                &k_v,
                &v_v,
                l as i32,
                pos as i32,
                kv_mul as i32,
                qk_scale,
            )
            .generics(generics)
            .sync()
            .unwrap();

            q_d = q_d.reshape(&[dim]).unwrap();

            // final matmul to get the output of the attention
            xb_d = xb_d.reshape(&[dim]).unwrap(); // flatten

            let x_p = (&mut x_d).partition([dim_ts]);

            let _ = kernel::fmatmul(x_p, &w.wo, &xb_d, l as i32)
                .first()
                .sync()
                .unwrap();

            // ffn rmsnorm
            let xb_part = (&mut xb_d).partition([dim_ts]);
            let (_, _, _, _) = kernel::rmsnorm(xb_part, &*x_d, &w.rms_ffn_weight, l as i32)
                .sync()
                .unwrap();

            // F.silu(self.w1(x)) * self.w3(x)
            let hd_ts = 64; // hidden dim tile size
            let d_ts = 64; // dim tile size;

            let hb_p = (&mut hb_d).partition([hd_ts]);

            let generics = vec![hd_ts.to_string(), d_ts.to_string()];
            let _ = kernel::matmul_silu(hb_p, &w.w1, &w.w3, &xb_d, l as i32)
                .generics(generics)
                .first()
                .sync()
                .unwrap();

            // x + self.w2(h)
            let x_part = (&mut x_d).partition([d_ts]);

            let generics = vec![d_ts.to_string(), hd_ts.to_string()];
            let (_, _, _, _) = kernel::matmul_ffn(x_part, &w.w2, &*hb_d, l as i32)
                .generics(generics)
                .sync()
                .unwrap();
        }

        // final rmsnorm
        let x_part = (&mut x_d).partition([dim.next_power_of_two()]);
        let (_, _) = kernel::rmsnorm_inplace(x_part, &w.rms_final_weight)
            .sync()
            .unwrap();

        // classifier into logits
        let vocab_ts = 32; // vocab_size tile size
        let d_ts = 64; // dim tile size;

        let logits_p = Arc::get_mut(logits_d).unwrap().partition([vocab_ts]);

        let generics = vec![vocab_ts.to_string(), d_ts.to_string()];
        let _ = kernel::matmul_cls(logits_p, &w.wcls, &*x_d)
            .generics(generics)
            .first()
            .sync()
            .unwrap();

        self.logits = (&*logits_d).to_host_vec().sync().unwrap();

        // restore kv cache
        s.xb_d = Some(xb_d);
        s.q_d = Some(q_d);

        // return
        &mut self.logits
    }
}
