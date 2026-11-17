# Code conventions

**Date:** 2026-09-30

Conventions for keeping the irap-tools codebase consistent, adapted from the conventions of Vidlu (`vidlu/docs/conventions.md`).

See [`../AGENTS.md`](../AGENTS.md) for broader coding guidelines that apply across projects, and the README of each package under [`../packages/`](../packages/) for what it does.

---

## 1. Naming

### 1.1 General (PEP 8)

- `PascalCase` for classes.
- `snake_case` for functions, methods, variables, modules, and file names.
- `UPPER_SNAKE_CASE` for module-level constants.
- Leading `_` for private module members and attributes.

Names used outside a small local scope should be specific: `compute_split_labels`, not `compute_labels`.

A boolean that describes a state should read as an affirmative statement, not a bare adjective or a question. An attribute completes the sentence begun by its owner (`self.is_selected`), so its name does not repeat the owner. A variable has no owner, so its name starts with its own subject (`target_is_set`).

A boolean argument that tells a function what to do is named as a command (`allow_missing_attributes=True`, `use_ncontext_filter=False`, `normalize=False`) and passed by keyword, so that call sites name each flag.

Name an identifier after the quantity it represents, not the symbol used for it in a formula: `learning_rate` over `alpha`, `std_dev` over `sigma`. Exceptions: conventional symbols (`x`, `y`) and short loop indices.

Prefix counts with `num_` rather than suffixing them with `_count`: `num_segments`, `num_missing`.

Reserve `eps` / `EPS` for a numerical tolerance that absorbs floating-point error or guards near-zero values. Use `threshold` only for a genuine decision boundary between two meaningful regimes.

### 1.2 Function naming by role

Use an **imperative verb phrase**, with exceptions:
- **Predicates** should be phrased as affirmative statements, with the same subject rule as boolean attributes ([§ 1.1](#11-general-pep-8)): `is_unlabeled_split()`, `contains()`, `intersects()`.
- If there is a **standard noun term** for a function, it is acceptable: `exp`, `argmax`, `fft`.
- **Conversions** with `to` or `from` as the prefix or infix can be named without a verb: `direction_to_heading`.
- **Factories and constructors** can be named after what they produce.
- **Event handlers** should be named with the `on_` prefix followed by the event name (a noun with a past participle): `on_iter_finished`.

**Accessors**: prefer a Python property (`@property`) whenever the value is cheap, deterministic, and side-effect-free. Otherwise use `get_`, or name the work, for example `fetch_` for I/O other than reading files.

| Prefix | Role | Example |
|---|---|---|
| `compute_` | Pure algorithm / math | `compute_split_labels` |
| `select_` | Choose a subset of segments | `select_split_segments` |
| `load_` | Read and parse files | `load_irap_metadata` |
| `write_` | Write files | `write_statistics_reports` |
| `make_` | Factory | `make_irap_data` |
| `set_` / `update_` | Mutator / partial state change | `set_lr`, `update_metrics` |
| `get_` | Accessor, possibly with args | `get_segment_location` |
| `run_` / `execute_` | Start a long-running process, e.g. a command | `run_report` |
| `to_` / `from_` | Type conversion | `to_html`, `to_json_dict` |
| `is_` / `has_` / `can_` | Boolean predicate | `is_unlabeled_split`, `AttributeDistribution.is_labeled` |

A method that doesn't read `self` should be a module-level function instead of an instance method.

### 1.3 Factory arguments – the `_f` suffix

A parameter whose value is a *callable that produces an object* (rather than the object itself) is suffixed with `_f`: `frame_converter_f`, `video_writer_f`, `pbar_f`.

This lets callers customize nested construction with `functools.partial`, without a parallel configuration structure:

```python
from functools import partial

iterate_video_frames(reader, frames=frame_indices, pbar_f=partial(tqdm, desc="Processing"))
```

### 1.4 Common type suffixes

Reuse established suffixes before inventing new ones:

| Suffix | Role |
|---|---|
| `*Dataset` | Subclass of `irap_data.dataset.Dataset` (e.g. `IRAPDataset`) |
| `*Statistics` | Computed statistics of a set of segments (e.g. `SplitStatistics`) |
| `*Preset` | Default settings of a release (e.g. `DatasetPreset`) |
| `*Config` | Configuration data class (e.g. `ViewerConfig`) |
| `*Error` | Exception class (e.g. `PredictionFormatError`) |

### 1.5 Module import aliases

Reuse the established alias rather than inventing a new one:

| Import | Alias |
|---|---|
| `typing` | `T` |
| `numpy` | `np` |
| `dataclasses` | `dc` |
| `pandas` | `pd` |
| `streamlit` | `st` |
| `irap_data.reports.document` | `rd` |

---

## 2. Package layout

Each tool is a package under `packages/` with its own `pyproject.toml`, README and tests:

```
packages/
├─ irap_data/                      Metadata API, PyTorch datasets, statistics reports, viewer
├─ irap_evaluation/                Evaluation of saved predictions (depends on irap-data)
├─ irap_evaluation_server/         Web app for submissions and their evaluation (NiceGUI)
├─ irap_video_cutting/             Cutting survey videos by GPS track
├─ irap_vietnam_360/               Fisheye to perspective conversion (superseded)
└─ irap_vietnam_data_preparation/  Scripts that build the iRAP-Vietnam metadata
```

An installable package uses the `src/` layout: `packages/<name>/src/<name>/`.

Group by domain subsystem, not by technical layer. Inside a package, the layers import only downward:

```
irap_data/
├─ attrs, metadata, lazy_dict   Metadata API (NumPy only)
├─ dataset, irap_dataset, …     PyTorch datasets (`torch` extra)
├─ reports/                     Statistics and their HTML reports, plots and maps
└─ tools/                       The command-line tool and the dataset viewer

irap_evaluation/
├─ predictions, evaluation, …   Prediction files, scoring and ensembling
├─ reports/                     Tables and JSON documents of results, coding tables
└─ tools/                       The command-line tool
```

The library imports neither `reports` nor `tools`, and `reports` does not import `tools`. The package `__init__` re-exports only the library. Place code in the lowest layer that uses it. For example, the dataset statistics are in `reports/`, because only the reports and the command-line tool use them.

An application package, such as a server, keeps its logic modules, UI and entry point at the top level. Its logic modules do not import the UI framework:

```
irap_evaluation_server/
├─ config, datasets, archive, …   Logic, tested without a browser
├─ components/                    Reusable UI parts
├─ pages/                         One module per page
└─ server                         The command that runs the app
```

Import an optional dependency (an extra, such as `torch`, `report` or `viewer`) only in the modules that need it, so that the rest of the package works without it. `import irap_data` does not import PyTorch or Matplotlib.

---

## 3. Module design

### 3.1 Values

Return computed results as frozen dataclasses (`@dc.dataclass(frozen=True)`), and document array fields with their shape and dtype.

Values that are derived from loaded files are `functools.cached_property` on the object that holds the files (for example `IRAPMetadata.sequence_index` and `IRAPMetadata.segment_coordinates`). A file that only some uses need is also read on first access.

### 3.2 Composition over parallel configuration

Prefer composing callables with `functools.partial` over introducing a separate configuration structure that mirrors a function's signature. A function that only forwards arguments to another takes them as `**kwargs` and names the function that receives them, instead of repeating its parameters and defaults.

---

## 4. Physical quantities and units

- Use SI internally. When the unit is ambiguous, encode it in the name with a snake_case suffix: `duration_s`, `interval_ms`, `angle_deg`.
- Use `YYYY-MM-DD` for dates in filenames, docstrings, and notes.

---

## 5. Documentation and prose

### 5.1 Writing style

This section applies to all prose in the repository: docstrings, comments, markdown documents, READMEs, and commit messages.

**Punctuation and casing.**

- Always use the spaced en-dash (` – `) – never use the em-dash (`—` / ` — `), and never a double hyphen (`--`) standing in for either.
- Avoid semicolons. Use a comma, a full stop, or an en-dash, whichever the sense calls for.
- Use sentence case for all titles and headings (for example, "How it works", "Dataset statistics report"), capitalising only the first word and proper nouns.

**Content and phrasing.**

- Convey meaning as efficiently as possible: give the essential information, and avoid verbosity, conversational filler, restating what the code plainly says, and redundancy with respect to other documentation.
- Avoid loose abbreviations in prose: "implementation", not "impl", and "configuration", not "config", as a noun.
- Use ISO 24495-1:2023 Plain language or ASD-STE100 Simplified Technical English (STE) where it does not detract from meaning. Consider also the Diátaxis framework for structuring information.
- **Single source of truth applies to comments too.** Document a rule once, at the type or function that owns it, and reference it elsewhere. At a call site, keep only what is local to that site – why *this* caller does what it does.
- Prefer plain, accurate language over metaphor and jargon. Use verbs that describe the actual operation (e.g. a tensor is sliced, a batch is collated, an event fires).
- **Name a thing with a noun.** For an operation referred to *as a thing*, use the noun or gerund English already provides, not the bare verb stem: "forward pass" (not "at forward"), "quantization" (not "the quantize"), "encoding" (not "the encode"). Established zero-derived nouns (a build, a call, a step) are exempt.
- Prefer a simple negation to an absolute or temporal adverb: "does not crop", not "never crops". Use temporal adverbs ("always", "never") only for temporal invariants the code enforces.
- **Describe current behaviour only.** After a rename or behaviour change, update affected comments in the same edit. Record historical context, such as what an earlier version did, in commit messages or development notes, not in code comments.

Where this section is silent, follow the [Google developer documentation style guide](https://developers.google.com/style).

### 5.2 Code documentation (docstrings and comments)

- Google-style docstrings (`"""..."""` with `Args:` / `Returns:` / `Raises:`) for every public class and for functions whose behaviour isn't obvious from the signature. Skip self-documenting methods.
- Document non-obvious parameter contracts: array shape (e.g. `(N, A)`), value range (e.g. `[0, 1]`), or convention (e.g. `IGNORE_LABEL_INDEX` for no label).
- Prefer a more descriptive identifier over an inline comment. Add a comment only for rationale a name cannot carry, and explain *why*, not *what*.
- Function docstrings should be descriptive rather than imperative statements (e.g. "Computes the confusion matrix across all batches.", not "Compute the confusion matrix").
- Use two spaces before an inline comment: `x = 1.0  # learning rate`.
- Use section banner comments for navigating long files:

  ```python
  # Section name ####################################################################################
  ```

### 5.3 Commit messages

Write the subject in the **imperative mood**, as [git-commit](https://git-scm.com/docs/git-commit#_discussion) asks, completing *"if applied, this commit will..."* (`Add`, `Move`, `Fix`, `Upgrade`, `Split`, `Remove`, `Refactor`), not *"if the commit is applied, the code will..."*.

**The verb names what the commit does to the codebase, not what the software does afterwards:** `Add CSV export of the dataset statistics`, not `Export the dataset statistics to CSV`.

- One line, capitalised, no trailing period, and no type prefix such as `feat:` / `fix:`.
- A full descriptive clause rather than a terse label: `Split image folders that hold several recordings when building the Vietnam metadata`, not `Folder fixes`.
- Add a body separated by a blank line for non-trivial changes, explaining what changed and why. Each paragraph is written on one line without hard-wrapping.
