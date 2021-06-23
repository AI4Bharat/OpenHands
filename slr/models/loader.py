import hydra

def load_encoder(encoder_cfg, dataset):
    if encoder_cfg.type == "cnn3d":
        from .encoder.cnn3d import CNN3D
        return CNN3D(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "cnn2d":
        from .encoder.cnn2d import CNN2D
        return CNN2D(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "decoupled-gcn":
        from .encoder.graph.decoupled_gcn import DecoupledGCN
        return DecoupledGCN(in_channels=dataset.in_channels, **encoder_cfg.params)
    else:
        exit(f"ERROR: Encoder Type '{encoder_cfg.type}' not supported.")

def load_decoder(decoder_cfg, dataset, encoder):
    if decoder_cfg.type == "fc":
        from .decoder.fc import FC
        return FC(n_features=encoder.n_out_features, num_class=dataset.num_class, **decoder_cfg.params)
    elif decoder_cfg.type == "rnn":
        from .decoder.rnn import RNNClassifier
        return RNNClassifier(n_features=encoder.n_out_features, num_class=dataset.num_class, **decoder_cfg.params)
    elif decoder_cfg.type == "bert":
        from .decoder.bert import BERT
        return BERT(n_features=encoder.n_out_features, num_class=dataset.num_class, config=decoder_cfg.params)
    else:
        exit(f"ERROR: Decoder Type '{decoder_cfg.type}' not supported.")

def get_model(config, dataset):    
    encoder = load_encoder(config.encoder, dataset)
    decoder = load_decoder(config.decoder, dataset, encoder)

    from .network import Network
    return Network(encoder, decoder)
