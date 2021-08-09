import omegaconf
import torch.nn as nn


def load_encoder(encoder_cfg, dataset):
    if encoder_cfg.type == "cnn3d":
        from .encoder.cnn3d import CNN3D

        return CNN3D(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "cnn2d":
        from .encoder.cnn2d import CNN2D

        return CNN2D(in_channels=dataset.in_channels, **encoder_cfg.params)

    #### GRAPH MODELS FOR POSE ####
    elif encoder_cfg.type == "pose-flattener":
        from .encoder.graph.pose_flattener import PoseFlattener

        return PoseFlattener(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "decoupled-gcn":
        from .encoder.graph.decoupled_gcn import DecoupledGCN

        return DecoupledGCN(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "st-gcn":
        from .encoder.graph.st_gcn import STGCN

        return STGCN(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "sgn":
        from .encoder.graph.sgn import SGN

        return SGN(in_channels=dataset.in_channels, **encoder_cfg.params)
    elif encoder_cfg.type == "gcn":
        from .encoder.graph.gcn import GCNModel
        from .encoder.graph.pose_flattener import PoseFlattener

        return nn.Sequential(
            GCNModel(in_channels=dataset.in_channels, **encoder_cfg.params),
            PoseFlattener(
                in_channels=dataset.in_channels,
                num_points=encoder_cfg.params.num_points,
            ),
        )
    elif encoder_cfg.type == "pretrained_encoder":
        # TODO: Directly load from .ssl.pretrainer
        from ..core.pretraining_model import PosePretrainingModel

        cfg = omegaconf.OmegaConf.load(encoder_cfg.params.cfg_file)
        pretrainer = PosePretrainingModel.load_from_checkpoint(
            encoder_cfg.params.ckpt,
            model_cfg=cfg.model,
            params=cfg.params,
            create_model_only=True,
        )
        return pretrainer.model.bert
    else:
        exit(f"ERROR: Encoder Type '{encoder_cfg.type}' not supported.")


def load_decoder(decoder_cfg, dataset, encoder):
    # TODO: better way
    if isinstance(encoder, nn.Sequential):
        n_out_features = encoder[-1].n_out_features
    else:
        n_out_features = encoder.n_out_features

    if decoder_cfg.type == "fc":
        from .decoder.fc import FC

        return FC(
            n_features=n_out_features, num_class=dataset.num_class, **decoder_cfg.params
        )
    elif decoder_cfg.type == "rnn":
        from .decoder.rnn import RNNClassifier

        return RNNClassifier(
            n_features=n_out_features, num_class=dataset.num_class, **decoder_cfg.params
        )
    elif decoder_cfg.type == "bert":
        from .decoder.bert_hf import BERT

        return BERT(
            n_features=n_out_features,
            num_class=dataset.num_class,
            config=decoder_cfg.params,
        )
    elif decoder_cfg.type == "fine_tuner":
        from .decoder.fine_tuner import FineTuner

        return FineTuner(
            n_features=n_out_features, num_class=dataset.num_class, **decoder_cfg.params
        )
    else:
        exit(f"ERROR: Decoder Type '{decoder_cfg.type}' not supported.")


def get_model(config, dataset):
    encoder = load_encoder(config.encoder, dataset)
    decoder = load_decoder(config.decoder, dataset, encoder)

    from .network import Network

    return Network(encoder, decoder)
