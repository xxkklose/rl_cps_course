import argparse
from algorithm.ppo import ppo_train

def load_cfg(path):
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        cfg = {}
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#') or ':' not in s:
                    continue
                k, v = s.split(':', 1)
                k = k.strip()
                v = v.strip()
                lv = v.lower()
                if lv in ('true','false'):
                    cfg[k] = lv == 'true'
                else:
                    try:
                        if '.' in v or 'e' in lv:
                            cfg[k] = float(v)
                        else:
                            cfg[k] = int(v)
                    except:
                        cfg[k] = v
        return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config_path = f'config/{args.config}.yaml'
    cfg = load_cfg(config_path)

    ppo_train(
        env_id=cfg.get('env_id', 'manipulator_mujoco/AuboI5Env-v0'),
        total_steps=int(cfg.get('total_steps', 100000)),
        rollout_len=int(cfg.get('rollout_len', 2048)),
        update_epochs=int(cfg.get('update_epochs', 10)),
        minibatch_size=int(cfg.get('minibatch_size', 256)),
        gamma=float(cfg.get('gamma', 0.99)),
        lam=float(cfg.get('lam', 0.95)),
        clip_ratio=float(cfg.get('clip_ratio', 0.2)),
        vf_coef=float(cfg.get('vf_coef', 0.5)),
        ent_coef=float(cfg.get('ent_coef', 0.0)),
        lr=float(cfg.get('lr', 3e-4)),
        device=str(cfg.get('device', 'cpu')),
        render=bool(cfg.get('render', False)),
        save_dir=str(cfg.get('save_dir', 'runs/ppo')),
    )

if __name__ == '__main__':
    main()
