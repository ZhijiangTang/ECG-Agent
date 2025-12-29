import hydra
from omegaconf import OmegaConf
from utils.utils import set_seed,count_parameters
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(config_path="config", config_name="validation", version_base=None)
def main(cfg):
    args = OmegaConf.to_container(cfg, resolve=True)
    set_seed(args['seed'])
    run_name = f"{args['model_name']}_{args['task']}_{args['dataset_name']}_{args['sub_dataset']}_finerate{args['finetune_rate']}_{args['dataset_balance']}_lr{args['learning_rate']}_frozen{args['is_frozen']}"
    args['run_name'] = run_name

    # TODO: more models
    # - GQRS
    # - SECGNet
    # - DENSECG
    # - Medformer
    # - Informer
    # - BaseLTM

    base_model = hydra.utils.instantiate(cfg.model)
    train_data = hydra.utils.instantiate(cfg.data.data, args.copy(), is_train=True)
    if args["data"]["task"] == "Classification":
        args["num_class"] = train_data.num_class
        model = hydra.utils.instantiate(cfg.data.head, args.copy(), base_model=base_model,  is_frozen=cfg.model.is_frozen, num_class=args['num_class'])
    else:
        model = hydra.utils.instantiate(cfg.data.head, args.copy(), base_model=base_model, is_frozen=cfg.model.is_frozen)
    args['model_size'] = count_parameters(model)
    test_data = hydra.utils.instantiate(cfg.data.data,args.copy(), is_train=False)

    val = hydra.utils.instantiate(cfg.data.validation, args.copy(), model, train_data, test_data)
    if args['is_load_finetune_model'] == 0:
        val.finetune()
    val.validation()
    summary = val.summary()
    if args['is_save_result']:
        val.save_result()
    
    val.sample_analysis(5)
    val.sample_analysis(10)
    val.sample_analysis(20)
    val.sample_analysis(30)


if __name__ == '__main__':
    main()
