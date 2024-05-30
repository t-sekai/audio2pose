import wandb
import uuid

class Logger:
    def __init__(self, args):
        if args.disable_wandb:
             self._wandb = None
             return
        
        self.project = args.wandb_project
        self.group = args.wandb_group
        self.name = args.random_seed
        self.entity = args.wandb_entity

        wandb.init(
            project=self.project,
            group=self.group,
            name = str(self.name),
            id=str(uuid.uuid4()),
            entity=self.entity,
            dir=args.root_path+args.out_root_path,
        )
        wandb.config.update(args)
        wandb.run.save()
        self._wandb = wandb
		
    def log(self, d, category="train"):
        if self._wandb:
            xkey = "iteration"
            _d = dict()
            for k, v in d.items():
                _d[category + "/" + k] = v
            self._wandb.log(_d, step=d[xkey])

    def finish(self):
        if self._wandb:
            self._wandb.finish()