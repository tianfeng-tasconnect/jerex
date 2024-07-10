import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import PredictConfig
from jerex import model, util

cs = ConfigStore.instance()
cs.store(name="predict", node=PredictConfig)


@hydra.main(config_name='predict', config_path='configs/docred_joint')
def predict(cfg: PredictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.dataset, 'predict_path')
    util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    model.predict(cfg)


if __name__ == '__main__':
    predict()
