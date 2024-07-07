class SequenceLAVAEncoder(nn.Module):
    """Full Sequence EBM."""

    image_encoder: str
    lang_encoder: str

    num_layers: int = 2

    sequence_length: int = 4
    temporal_transformer_num_layers: int = 2

    d_model: int = 128
    num_heads: int = 2
    pyramid_fuse_layers: Tuple[int, Ellipsis] = (2, 3, 4)

    @nn.compact
    def __call__(self, x, *, train):
        """Apply the ResNet to the inputs `x`.

        Args:
          x: Inputs.
          train: Whether to use BatchNorm in training or inference mode.

        Returns:
          The output head with `num_classes` entries.
        """
        rgb = x["rgb"]
        print(f"Initial rgb shape: {rgb.shape}")
        bs = rgb.shape[0]
        seqlen = rgb.shape[1]
        h = rgb.shape[2]
        w = rgb.shape[3]
        c = rgb.shape[4]
        rgb = jnp.reshape(rgb, (bs * seqlen, h, w, c))
        print(f"Reshaped rgb shape: {rgb.shape}")

        if self.image_encoder == "resnet":
            features = ResNetVisualEncoder()(rgb, train=train)
        elif self.image_encoder == "conv_maxpool":
            features = ConvMaxpoolCNNEncoder()(rgb, train=train)
        else:
            raise NotImplementedError

        visual_sentence = VisualDescriptorsNet(
            d_model=self.d_model, pyramid_fuse_layers=self.pyramid_fuse_layers)(
                features, train=train)
        visual_sentence = nn.Dropout(0.1)(visual_sentence, deterministic=not train)

        if self.lang_encoder == "clip_in_obs":
            lang_embedding = x["clip_embedding"]
            lang_embedding = jnp.reshape(lang_embedding, [bs * seqlen, -1])
        elif self.lang_encoder == "clip":
            tokens = x["instruction_tokenized_clip"].astype(jnp.int32)[:, 0]
            lang_embedding = clip_layers.TextEncoder(
                vocab_size=49408,
                features=512,
                num_heads=8,
                num_layers=12,
                out_features=512,
            )(
                tokens)
            lang_embedding = jnp.tile(lang_embedding[:, None, :], [1, seqlen, 1])
            lang_embedding = jnp.reshape(lang_embedding, [bs * seqlen, -1])
            lang_embedding /= jnp.linalg.norm(lang_embedding, axis=-1, keepdims=True)
        else:
            raise NotImplementedError

        normal_initializer = jax.nn.initializers.normal(stddev=0.05)

        lang_embedding = nn.Dense(
            self.d_model,
            kernel_init=normal_initializer,
            bias_init=normal_initializer)(
                lang_embedding)

        lang_embedding *= jnp.sqrt(self.d_model)  # scale emb.
        lang_embedding = nn.Dropout(0.1)(lang_embedding, deterministic=not train)

        lang_embedding = lang_embedding[:, None, :]
        lang_query = lang_embedding

        fused_x = lang_query
        for _ in range(self.num_layers):
            fused_x = PrenormPixelLangEncoder(
                num_heads=2, dropout_rate=0.1, mha_dropout_rate=0.0,
                dff=self.d_model)(
                    visual_sentence, fused_x, train=train)

        fused_x = jnp.squeeze(fused_x, axis=1)

        fused_x = nn.LayerNorm()(fused_x)

        seq_obs_encoding = jnp.reshape(fused_x, [bs, seqlen, -1])

        obs_encoding = TemporalTransformer(
            num_layers=self.temporal_transformer_num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.d_model,
            sequence_length=self.sequence_length)(
                seq_obs_encoding, train=train)
        return obs_encoding
