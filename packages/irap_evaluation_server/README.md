# iRAP evaluation server

A web app that keeps an archive of model predictions, scores them, and compares, analyses and ensembles them. It is built with [NiceGUI](https://nicegui.io) on [`irap_evaluation`](../irap_evaluation/).

| Page | Path | Content |
|---|---|---|
| Scores | `/scores` (start page) | The methods of a dataset split ranked by an attribute average, or one per-attribute metric, with intervals and differences from a reference method. |
| Models | `/models` | The models of each dataset by method, with the file of each split, a filter, the upload form, and the form that creates an ensemble of the models selected in the tables. |
| Model | `/models/<id>` | A model's files, its scores, the coding-table export and its actions (collapsed, for accounts with write permission). |
| Method | `/methods?dataset=<dataset>&method=<method>` | A method's models, the scores of their mean and of each model on a split, its display name and description, and the actions of its models. Method names in the tables link to it. |
| Analysis | `/analysis` | A model's confusion matrix and review curve for one attribute, a map of its outcomes, and segment details, optionally compared with a second model. |
| Methodology | `/methodology` | How the files are scored, and how to read the values, intervals, highlights and differences of the score tables. The short notes below the tables link to its sections. |
| Action log | `/actions` | All changes, with the name of the user. Only for accounts with write permission. |
| Accounts | `/accounts` | The accounts, whose permissions admins change. Only for admins, in the Admin menu. |
| Shown metrics | `/shown-metrics` | The attribute averages that the pages show, which admins choose. Only for admins, in the Admin menu. |

Anyone who can open the app can view it and download files. Only signed-in [accounts](#accounts) with write permission can upload, replace and delete files, delete models and methods, edit descriptions and method display names, and create ensembles, and see the action log, which shows the account name of each change.

## Installation and start

From the irap-tools checkout, install the packages and start the server with a [configuration file](#configuration):

```bash
uv pip install -e packages/irap_data -e packages/irap_evaluation -e packages/irap_evaluation_server
irap-eval-server server.toml
```

The map of the Analysis page needs internet access:

- On its first start, the server downloads MapLibre and deck.gl (pinned versions checked by SHA-256) into `data_dir/vendor/`, since MapLibre must be served by the server itself. Without internet access, it starts without the map.
- The browser loads the basemap tiles from Carto.

It also downloads MathJax 3 (`tex-svg.js`, pinned and checked the same way), which typesets the formulas of the Methodology page. Without it, the page shows them as TeX.

## Configuration

`server.toml` (relative paths are relative to its directory):

```toml
data_dir = "data"       # the archive
host = "127.0.0.1"      # optional, "0.0.0.0" to serve the lab network
port = 8600             # optional
is_served_over_https = false   # optional, true behind an HTTPS proxy: the session cookie is then sent only over HTTPS

[datasets.vietnam]      # a key of irap_data.DATASET_PRESETS
dataset_dir = "/data/IRAP_Vietnam"   # the images (FRAMES/) and the metadata
analysis_splits = ["train", "val"]   # the splits of the Analysis page, e.g. without test to keep its labels unseen; the others are protected

[datasets.bh]
dataset_dir = "/data/IRAP_BIH"
metadata_dir = "/data/IRAP_BIH_METADATA"   # optional, default dataset_dir
analysis_splits = ["train", "val"]
```

The image paths in the metadata (`segment_id_to_data_paths_rel.json`) are relative to `dataset_dir`. A Vietnam dataset directory, as [`irap_vietnam_data_preparation`](../irap_vietnam_data_preparation/README.md#directory-layout) builds it, also holds the metadata. Without the images, e.g. on a server with only the metadata files, the Analysis page shows no images.

The data directory holds:

- `archive.sqlite3`: the index of models and files, and the action log,
- `submissions/<id>/predictions.parquet`: the stored files, removed when they are [deleted](#deletion) (or at a later deletion if a file is open, e.g. on Windows),
- `scores.sqlite3`: the [scores](#scoring), which the server computes again from the files if the file is missing, so backups can leave it out,
- `uploads/`: uploaded files that are not stored yet, removed at start, and when another upload begins if they are older than 24 h,
- `vendor/`: the map libraries and MathJax,
- `accounts.sqlite3`: the [accounts](#accounts),
- `display_settings.sqlite3`: the [hidden metrics](#hidden-metrics),
- `nicegui_storage/` and `storage_secret.txt`: NiceGUI's per-browser storage, e.g. of the signed-in account, and the key of its session cookie. Keep them, and backups of the data directory, private. Deleting the key signs out all users.

The databases have a schema version. At start, the server migrates an archive, accounts or display settings database of an older version, after copying it to a backup next to it, e.g. `archive.v1.2026-10-12T150000Z.backup.sqlite3`. It refuses a database of a newer version, or one whose tables differ from those of its version. A score cache of another version is created again, and all files are scored again, which takes a few minutes, during which the Scores page lists the files that are being scored.

## Accounts

Users register and sign in with the links in the top bar. An account has one of the permissions:

| Permission | Can |
|---|---|
| view | View, like visitors without an account. |
| write | Also upload, replace and delete files, delete models and methods, edit descriptions and method display names, and create ensembles. |
| admin | Also download the per-class scores of [protected splits](#protected-splits), change the permissions of the accounts and remove them, on the Accounts page, and choose the [shown metrics](#hidden-metrics). |

The first account to register is an admin. Later accounts can view until an admin gives them more. The app keeps at least one admin: it refuses to remove the last admin or to take away its admin permission.

`accounts.sqlite3` has scrypt hashes of the passwords. Account names are unique regardless of the case of ASCII letters. A permission change applies at once, also to accounts that are signed in. A session lasts until 14 days after the browser's first visit, also if it signed in later, or until the account is removed. Every change is checked on the server when it is made, also if a page was opened before a sign-out or a permission change.

There is no password change. An admin can remove an account with a lost password, so that the user can register the name again. If no admin can sign in any more, stop the server and delete `accounts.sqlite3`, which removes all accounts, so that the next to register becomes the admin. The action log keeps the names.

The app does not limit registrations or sign-in attempts. Use long passwords. See [Public deployment](#public-deployment) for HTTPS.

### Protected splits

The splits of a dataset that are not in its `analysis_splits`, e.g. test, are protected: their labels are not shown in detail. Everyone can store their files and see their per-attribute and averaged scores, but they are not on the Analysis page, and only admins get their per-class scores, confusion matrices and review curves, which are left out (`"classes": null`) of the scores JSON that others download. For the same reason, only admins see the Class count, Minority-class example count and Smallest-class example count columns of the per-attribute view of the Scores page. The number of labeled segments of each attribute (Example count) is shown to everyone, as it is in the scores JSON (`metrics.per_attribute.n`).

This does not keep the labels secret from writers, since the scores of chosen predictions can reveal labels. Give write permission only to trusted users. The action log shows who stored which file.

## Public deployment

On a server with a public DNS name, the app listens only on `127.0.0.1`, and [Caddy](https://caddyserver.com) serves it over HTTPS. The files are in [`deployment/`](deployment/). The steps are for Arch Linux.

1. Install the packages, create a system user, and install the app in a virtual environment:

   ```bash
   sudo pacman -S --needed caddy git uv
   sudo useradd --system --create-home --home-dir /srv/irap-eval --shell /usr/bin/nologin irap-eval
   sudo -u irap-eval -H bash -c 'cd ~ && mkdir -p data \
       && git clone https://github.com/Ivan1248/irap-tools.git && uv venv venv \
       && VIRTUAL_ENV=~/venv uv pip install -e irap-tools/packages/irap_data \
          -e irap-tools/packages/irap_evaluation -e irap-tools/packages/irap_evaluation_server'
   ```

2. Copy [`deployment/server.toml`](deployment/server.toml) to `/srv/irap-eval/server.toml` and set the dataset directories in it:

   ```bash
   sudo -u irap-eval cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/server.toml /srv/irap-eval/
   ```

   The user `irap-eval` must be able to read the dataset directories. The service cannot read `/home` (`ProtectHome`), so the datasets must be elsewhere, e.g. under `/data`.

3. Start the app as a service:

   ```bash
   sudo cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/irap-eval-server.service /etc/systemd/system/
   sudo systemctl enable --now irap-eval-server
   journalctl -u irap-eval-server -f   # the log
   ```

4. Register the first account, which becomes the admin, before the app is public. On your computer, forward a port to the app over SSH and open `http://localhost:8600/register`:

   ```bash
   ssh -L 8600:127.0.0.1:8600 <server>
   ```

   With `is_served_over_https = true`, the browser may not keep the sign-in over plain HTTP, but the account is registered.

5. Copy [`deployment/Caddyfile`](deployment/Caddyfile) to `/etc/caddy/Caddyfile`, with the server's name in it, and start Caddy:

   ```bash
   sudo cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/Caddyfile /etc/caddy/Caddyfile
   sudo systemctl enable --now caddy
   journalctl -u caddy -f   # shows whether the certificate was issued
   ```

   Caddy gets a Let's Encrypt certificate for the name and renews it. Let's Encrypt must reach the server on port 80 or 443. Caddy also redirects HTTP to HTTPS.

6. Open only SSH, HTTP and HTTPS in the firewall, e.g. with `ufw`:

   ```bash
   sudo pacman -S ufw
   sudo ufw allow ssh && sudo ufw allow http && sudo ufw allow https
   sudo ufw enable && sudo systemctl enable ufw
   ```

   A network firewall of the institution may also block incoming connections.

To update the app, also to the newest versions of its dependencies, e.g. with security fixes:

```bash
sudo -u irap-eval -H bash -c 'cd ~/irap-tools && git pull && cd ~ \
    && VIRTUAL_ENV=~/venv uv pip install --upgrade -e irap-tools/packages/irap_data \
       -e irap-tools/packages/irap_evaluation -e irap-tools/packages/irap_evaluation_server'
sudo systemctl restart irap-eval-server
```

Back up `/srv/irap-eval/data`, and keep the backups private.

## Models and files

A model is, e.g., one trained network, identified by the dataset, method name and seed in the header of its [prediction files](../irap_evaluation/docs/prediction_format.md). Unlike `irap_evaluation`, the app requires a seed also for the only model of a method, so that a second model can be added without changing the first. It has at most one file per split. The pages call a stored file "file #<id>", and the code calls it a submission (`archive.Submission`).

A model has a description in Markdown, e.g. how it was trained, whose first line the tables show, so it should summarize the model. The Model page shows and edits it, and the Method page shows that of the model with the lowest seed and edits it for all models of the method. The Markdown is sanitized in the browser (DOMPurify), so scripts and event handlers in it are removed.

A method on a dataset can have a display name, which the pages show instead of the method name. The Method filter of the Scores page and the Model filter of the Models page match the shown name. An upload sets the display name of its method, entered in the upload form or from a file header's `method_display_name`, and the Method page edits it for all models of the method. The action log records the edit on the method's first model. Files, URLs, download names and the action log keep the method name.

### Uploads

A file is refused if:

- it has more than 1 GiB, or the files of a `.zip` archive have more than 1 GiB together,
- it is not a valid prediction file,
- it has no seed, neither in its header nor in the Seed field,
- its dataset is not configured, or its splits, attributes or iRAP codes are not those of the dataset,
- it lacks segments of its [evaluation sets](../irap_evaluation/docs/evaluation.md#evaluation-sets),
- its output kind, context offsets, or training or early stopping splits differ from those of the model's other files,
- the method's models would not have distinct seeds or would differ in training or early stopping splits (`irap_evaluation.check_method_models`), since a method is one training recipe and its models are averaged.

Files of unlabeled splits are accepted but not scored.

- The Seed field replaces the seed in the header, e.g. to add another model of a method, or gives a seed to a file without one. The action log records the original seed.
- A file of a split that the model already has replaces its file, which is deleted permanently, with its scores (see [Deletion](#deletion)). The action log records the deletion.
- A blank description keeps the model's description.
- A method display name entered with the upload replaces the method's, without changing the files. A blank one takes the display name from the files, and files without one keep the method's. The files of one upload must then not have different display names. When a file is uploaded, the field's placeholder shows the display name that a blank one gives.
- The user confirms the deletion of replaced files, and a replaced description or method display name before anything is stored. If the model changes in the meantime, e.g. by another user, the upload is refused and can be retried.

A `.zip` archive uploads the files of one model on several splits at once. Every member must be a `.parquet` prediction file (others are refused, not skipped), of the same model, output kind, context offsets, and training and early stopping splits (`archive.MODEL_FILE_FIELDS`), and of distinct splits. Their header `details` may differ. Either all files are stored, each as its own submission, or none. An archive that replaces all files of a model may change their output kind, context offsets, and training and early stopping splits.

### Deletion

Deletion is permanent. The Model page deletes a file, or the whole model with all its files, and the Method page deletes a whole method with all its models, after a confirmation. An upload or an ensemble also deletes the files that it replaces, after a confirmation. A deleted file is removed with its scores, and it cannot be restored, only uploaded again. The last file of a model is deleted only with the model, so every model has a file. Deleting the last model of a method also removes the method's display name. The action log keeps the actions on deleted models and files, and their ids are not reused.

To keep a copy, the Model page downloads the files of a model, e.g. an ensemble, as one `.zip` archive (`/api/models/<id>/predictions.zip`), which can be uploaded again.

## Scoring

A background thread scores each new file on its evaluation sets, with the defaults of `irap-eval evaluate` and 1000 bootstrap resamples (`scoring.ScoringSettings`). The attribute predictions of attributes that a file does not predict count as invalid (`--missing-attribute-policy invalid`), so all models are scored on the same attributes.

`scores.sqlite3` stores the attribute averages and per-attribute values, and their values in each bootstrap resample of road sequences (`irap_evaluation.resample_model_metrics`), as float32 (about 1.9 MB per file and evaluation set on Vietnam). Resampling takes about 1.1 s per file and evaluation set on Vietnam val and 1.5 s on train, most of it for the [manual review proportion](../irap_evaluation/docs/evaluation.md#manual-review-proportion), whose statistics also take about 80 MB of memory while a train file is scored. All files on an evaluation set share the resamples, so the pages compute means over models, intervals and paired comparisons of methods from the stored values, also over any subset of attributes, as `irap-eval evaluate` and `irap-eval compare` compute them (see [confidence intervals](../irap_evaluation/docs/evaluation.md#confidence-intervals)).

Stored scores are out of date when the scoring settings or the evaluation sets (e.g. after a metadata update) change. To catch changes of the scoring code, the settings fingerprint also includes the scores and resampled values of random predictions on a small synthetic set (2 attributes, 3 classes, 24 segments), computed at start. A change of `irap_evaluation` or NumPy that changes these values, e.g. of the random generator, makes all scores out of date, but one that changes only cases outside the set, e.g. attributes with more classes, does not. After such a change, increase `scoring.SCORE_CACHE_VERSION`, or delete `scores.sqlite3` while the server is stopped.

At start, the server rescores the files whose scores are out of date or failed with an unexpected error. A file that `irap_evaluation` refuses, e.g. one that lacks segments of a new metadata build, is marked as failed with the reason.

## Scores

The Scores page shows the means over each method's models with their bootstrap intervals.

- A value is bold if it is the best of its column (or row, in the per-attribute view), or not significantly worse than it: a one-sided paired bootstrap test at 5 %, whose 90 % interval of the difference from the best reaches 0 (`method_ranking.is_not_significantly_worse`). The test needs the methods on the same segments, so on the model-compatible sets values are bold only if all shown methods have the same context offsets.
- Methods trained on the shown split are hidden by default, since their scores are not of held-out data. Methods that used the split for early stopping, or whose training splits are unknown, are marked.
- `aMRP90` is the mean over the attributes of the [manual review proportion](../irap_evaluation/docs/evaluation.md#manual-review-proportion) `MRP90`: the fraction of an attribute's segments that a reviewer has to correct, starting with the least confident predictions, until its mF1 reaches 0.9. Like NLL and Brier, it needs probabilistic predictions, so methods of hard predictions have no value.
- The per-attribute view shows, before the methods, counts of each attribute's labels in the evaluation set (`label_counts`): Class count (the classes that occur as a label), Example count (the labeled segments), Minority-class example count (the labeled segments outside the class with the most) and Smallest-class example count (the labeled segments of the rarest of these classes). They are computed from the labels, not from the scores, since the scores' counts (`n`, `support`) leave out invalid predictions under `null_policy='exclude'`. On the model-compatible sets, they are shown only if all shown methods have the same context offsets.
- In the per-attribute view, a click on the header of a method or a label count, e.g. on the arrow below it, sorts the attributes by that column: worst first for a method, smallest first for a label count, and the reverse after a second click. A click on Attribute undoes the sorting. The attributes cannot be sorted by a hidden label count, since their order would reveal it.
- The Method filter is a case-insensitive regular expression that matches any part of the name, e.g. `foo`, or `^(foo|bar)$` for exactly these two.
- On the reference set and the full set, which are the same for all files of a split, "Compare with" adds each method's difference from a reference method, with its interval, as `irap-eval compare` computes it. See [comparing methods](../irap_evaluation/docs/evaluation.md#comparing-methods) for how to read them.
- The attribute filter, here and on the Method and Model pages, chooses the attributes that the averages cover. The averages of a method over a new subset take about 12 ms on Vietnam (35 attributes). The pages compute them in a thread, so that a large table does not hold up the server, and the worker keeps them in memory (`scoring_worker.NUM_CACHED_METHOD_RESAMPLES`).

The URL keeps the view, so it can be shared as a link. A method name links to the Method page with the same split and attributes. It shows the attribute averages of the method's mean and of each of its models, which are shaded but not tested against each other, and the per-attribute metrics of the mean. The Model page shows the scores of a file on each evaluation set and downloads them as the JSON document of `irap-eval evaluate`, without intervals (`/api/submissions/<id>/scores/<set>.json`). For a [protected split](#protected-splits), only admins get its per-class scores (`classes`).

The per-attribute tables of the Method and Model pages have mF1, mP, mR, MCC, the accuracy A and `MRP90`, and the number of invalid predictions. A is next to `MRP90` because the manual review proportion follows the error rate, 1 − A, more than mF1. A click on a column header sorts the attributes by that column, as in the per-attribute view of the Scores page: worst first for a metric, largest first for the invalid predictions, and the reverse after a second click. A click on Attribute undoes the sorting.

### Hidden metrics

On the Shown metrics page, admins hide attribute averages from the pages, all but the main metric, amF1. A hidden average also hides its per-attribute metric, e.g. aBrier hides Brier: from the columns, the sort and View options of the Scores page, the tables of the Method and Model pages, and the metric list of the Methodology page. A link with a hidden metric (`sort` or `metric`) is refused. The files are scored as before, and the downloaded scores have all metrics, so a metric that is shown again appears at once. Pages that are open show a change after a reload.

## Analysis

The Analysis page shows the predictions of a model (A) for one attribute on an evaluation set of an analysis split. Tabs switch between the confusion matrix with its class table, the class analysis and the review curve, and the URL keeps the choice:

- **Confusion matrix:** labels (rows) × predictions of A (columns), with a column of invalid predictions. Scores count an invalid prediction as the first class (`null_policy="first_class"`), i.e. they add the invalid column to the first.
- **Class table:** the support, precision, recall and F1 of each class, and how well A tells the class from the other classes of the attribute, one vs. rest: the AUROC of its predicted probability and the informedness J = TPR − FPR of its predicted classes (`class_analysis.compute_class_discrimination`). By chance they are 0.5 and 0 at any class size, unlike F1, which a predictor that ignores the image already gets between p and 2p / (1 + p) for a class of proportion p. A high AUROC with a J near 0 is a class that A ranks well but rarely predicts.
- **Class analysis:** the classes with labeled segments of all selected attributes that A predicts (`class_analysis`), for finding the classes that are hard for A: their support, F1, AUROC and J, the number of their segments that are predicted as another class (Missed), and the class that most of these are predicted as. By default, the classes with the lowest AUROC are first, and a click on a column header sorts by that column. A click on a class shows its attribute and selects its missed segments, as a click on its row label in the confusion matrix does.
- **Review curve:** mF1 and accuracy of A, and of B, after a reviewer has corrected a proportion of the segments, starting with the least confident predictions, with the point where mF1 reaches 0.9 (the manual review proportion `MRP90`). A tooltip shows the scores at each 5 % of the segments. Models of hard predictions have no curve.
- **Segments of a cell:** a click on a cell of the matrix lists its segments, most confident first, so that confident errors come first. A click on a row label of the matrix, or on a class of the class analysis, lists the missed segments of the class with their predicted classes, from the most confident errors to the invalid predictions.
- **Map:** segments coloured by outcome, or, with a second model (B), by which of A and B is correct. The listed segments are highlighted.
- **Segment details:** a click on a segment shows its context images (at A's context offsets, from `dataset_dir`), its label, and the distributions predicted by A and B.

The page keeps the 8 most recently used files in memory. Reading and aligning a file takes about 0.06 s on Vietnam val and 0.3 s on train.

## Ensembles

The "Create an ensemble" form of the Models page creates the weighted mean of the distributions of stored models of a dataset, as `irap-eval ensemble` does (`irap_evaluation.ensemble_predictions`). A hard prediction counts as a one-hot distribution. The members are the models selected with the checkboxes of the tables, which the form lists with their weights, also those that the filter hides.

- The ensemble is a model with the given method name, display name and seed (0 in a new form). A blank method name is made of the sorted method names of the members with their seeds, e.g. `strong-probs_seeds1-3` (`irap_evaluation.make_ensemble_method_name`). A blank display name keeps the method's.
- It gets a file for each split where every member has a file, and these replace all files of an existing model of that name, which are deleted, so that all its files are of the same members.
- Its training splits are the union of its members', and its early stopping splits are its members' that are not training splits (`irap_evaluation.make_ensemble_model_info`).
- Members that predict different segments or attributes are refused by default. The form can instead combine those that every member predicts, or all that any member predicts, where a member that does not predict one counts as invalid (`irap_evaluation.MemberCombination`). Members with different context offsets predict different segments. With all attributes, an attribute that only one member predicts is that member's prediction.
- Its files are checked, confirmed and scored like uploads, and the action log records the members, weights and combinations of segments and attributes. If a member's file changes before the ensemble is stored, the ensemble is refused and can be created again.

## Coding tables

The Model page downloads the files of the chosen splits as one iRAP coding table (`.xlsx` or `.csv`), as `irap-eval export-coding-table` does, with a coder name (default: the shown method name) and a coding date. An `.xlsx` table of probabilistic predictions has the probability of each exported code on a second sheet.

One table is made at a time, in a separate process (NiceGUI's `run.cpu_bound`, started with "spawn"), since an `.xlsx` file of large splits takes tens of seconds, which in a thread delayed the server's other pages until browsers lost the connection. The file is written to `<data_dir>/uploads` and sent once over HTTP. File names are cut to 160 bytes, since Excel does not open a file whose path is longer than 218 characters.

## Development

```
irap_evaluation_server/
├─ config.py                ServerConfig, DatasetConfig, load_server_config
├─ datasets.py              DatasetContext: the metadata of a dataset and its evaluation sets
├─ database.py              Connections to the SQLite databases and the migrations of their schemas
├─ archive.py               ModelArchive: models, their files, the SQLite index and action log
├─ accounts.py              AccountStore: the accounts, their permissions and password hashes
├─ uploads.py               The checks of uploaded files and .zip archives, and their planned updates
├─ model_listing.py         The models of the Models and Method pages, grouped by method
├─ scoring.py               Scoring settings and the score cache
├─ method_scores.py         Means over the models of a method and their resampled values
├─ scoring_worker.py        The background thread that scores, and the resampled values in memory
├─ method_ranking.py        The rows of the method table, their order, highlights and marks of used splits
├─ method_comparison.py     Comparisons with a reference method
├─ methodology.py           The text of the Methodology page, with the values of the settings
├─ prediction_analysis.py   Outcomes of segments, confusion matrices, segment details
├─ vendor_libraries.py      The download of the libraries of the map and of MathJax
├─ ensembles.py             Ensembles of stored models
├─ coding_table_export.py   The coding table of the files of a model
├─ components/              Parts of the pages, e.g. the page frame, form controls, tables and the map
├─ pages/                   One module per page
└─ server.py                The irap-eval-server command
```

The logic modules (all but `components`, `pages` and `server`) do not import NiceGUI and are tested with pytest:

```bash
uv run --project packages/irap_evaluation_server --extra test python -m pytest packages/irap_evaluation_server/tests
```

A data directory is in production since 2026-10-12, so a change of a database schema needs a new schema version and a migration from the previous one: `SCHEMA_VERSION` and `_MIGRATIONS` in `archive.py`, `accounts.py` and `display_settings.py` (see `database.initialize_schema`). A change of the score cache tables needs only a new `scoring.SCORE_CACHE_VERSION`. The server refuses tables that differ from those of their version, so a forgotten version shows at the next start.

[`scripts/add_synthetic_predictions.py`](scripts/add_synthetic_predictions.py) fills an archive with synthetic Vietnam predictions and those of an ensemble, e.g. to try the pages:

```bash
python packages/irap_evaluation_server/scripts/add_synthetic_predictions.py server.toml
```
