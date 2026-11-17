# Test deployment on strider (2026-10-05)

A temporary public deployment of `irap_evaluation_server` at https://strider.zemris.fer.hr: the app in a rootless Podman container in tmux, Caddy on the host for HTTPS. Nothing restarts after a reboot.

## Layout on strider

- `~/projects/irap-tools`: the checkout.
- `~/data/irap-eval/server.toml`: the configuration. The data directory `~/data/irap-eval/data` is created at the first start.
- `~/data/datasets/*`: symlinks to `/mnt/sdb1/datasets/*`. The container mounts `/mnt/sdb1/datasets` itself, since the absolute links do not resolve inside it.

## Installation

```bash
cd ~/projects/irap-tools
podman build -t irap-eval-server -f packages/irap_evaluation_server/deployment/Containerfile .
sudo pacman -S --needed caddy
```

## Setup

1. Create `~/data/irap-eval/server.toml` (the paths are those inside the container) and `chmod 700 ~/data/irap-eval`:

   ```toml
   data_dir = "data"
   host = "0.0.0.0"               # inside the container; published only on the host's 127.0.0.1
   port = 8600
   is_served_over_https = true

   [datasets.vietnam]
   dataset_dir = "/datasets/IRAP_Vietnam"
   analysis_splits = ["train", "val"]

   [datasets.bh]
   dataset_dir = "/datasets/IRAP_BIH"
   metadata_dir = "/datasets/IRAP_BIH_METADATA"
   analysis_splits = ["train", "val"]
   ```

2. Configure Caddy:

   ```bash
   sudo cp ~/projects/irap-tools/packages/irap_evaluation_server/deployment/Caddyfile /etc/caddy/Caddyfile
   sudo sed -i 's/eval\.example\.org/strider.zemris.fer.hr/' /etc/caddy/Caddyfile
   ```

## Starting

1. Start the app in tmux:

   ```bash
   tmux new -s irap-eval
   podman run --rm -it --init -p 127.0.0.1:8600:8600 \
       -v ~/data/irap-eval:/irap-eval \
       -v /mnt/sdb1/datasets:/datasets:ro \
       irap-eval-server /irap-eval/server.toml
   # Ctrl-b d detaches, tmux attach -t irap-eval reattaches, Ctrl-C stops the app
   ```

2. First start only: register the first account (the admin) before Caddy makes the app public, over a tunnel from the local computer, at `http://localhost:8600/register`:

   ```bash
   ssh -L 8600:127.0.0.1:8600 strider.zemris.fer.hr
   ```

3. Start Caddy:

   ```bash
   sudo systemctl start caddy     # not enabled
   journalctl -u caddy -f         # first start: "certificate obtained successfully"
   ```

## Update and shutdown

- Update: `git pull`, rebuild the image, stop the app with Ctrl-C and run `podman run` again.
- Shutdown: Ctrl-C in tmux, then `sudo systemctl stop caddy`.
