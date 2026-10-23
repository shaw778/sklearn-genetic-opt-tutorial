# sklearn-genetic-opt: Community Contribution & Technical Article Guide

Welcome! This workspace contains the complete end-to-end package for completing a **Documentation and Community Content contribution** to the [`sklearn-genetic-opt`](https://github.com/rodrigo-arenas/sklearn-genetic-opt) open-source repository.

---

## 🎯 Why This Contribution?

This task addresses open issue **#267 (`[DOCS] Share community-written articles and tutorials`)** under the repository's standing community program.

It gives you:
1. **A Published Technical Article**: Demonstrating your ability to explain complex machine learning concepts, evolutionary algorithms, and practical performance benchmarks.
2. **An Official Open-Source Merged Pull Request**: Added directly to the maintainer's repository on your GitHub profile without needing complex C/Cython/Python unit tests or core codebase architecture modifications.

---

## 📁 Workspace Contents

| File | Purpose |
| :--- | :--- |
| [`article.md`](./article.md) | **The complete publication-ready article**. Formatted in Markdown, ready to copy-paste into Dev.to, Medium, or a personal blog. |
| [`tutorial_code.py`](./tutorial_code.py) | **Executable verification script**. Validates the exact code, runs `GASearchCV` with multi-type search spaces, triggers `ConsecutiveStopping`, and outputs the benchmark comparison. |
| [`articles_patch.md`](./articles_patch.md) | **The exact Markdown snippet** to append as entry #8 to `docs-vitepress/versions/latest/community/articles.md`. |
| [`pull_request_template.md`](./pull_request_template.md) | **Pre-filled GitHub PR template** (title, description, checklist) aligned with merged community PRs (e.g. PR #379). |

---

## 🚀 The Two-Step Contribution Workflow

### Step 1: Publish the Article Externally

You can publish the contents of [`article.md`](./article.md) on any public platform under your own name.

#### Recommended Platforms:
1. **[Dev.to](https://dev.to)** *(Fastest & Recommended)*:
   - Free, developer-focused, instant publishing.
   - Fully supports native Markdown: click "Create Post", paste [`article.md`](./article.md), add tags (`#python`, `#datascience`, `#machinelearning`, `#opensource`), and click "Publish".
2. **[Medium](https://medium.com)**:
   - Click "Write a story", paste the content, format headers, add cover image, and publish.
3. **Personal Blog / GitHub Pages**:
   - Add to your Jekyll, Hugo, Astro, or Docusaurus site.

> 💡 *Once published, copy your live article URL (e.g., `https://dev.to/yourname/mastering-search-spaces-and-callbacks-in-sklearn-genetic-opt-xyz`).*

---

### Step 2: Submit Your Pull Request

Once your article is live, submit a tiny pull request to `sklearn-genetic-opt`.

#### Method A: Via GitHub Web Interface (Simplest, No Local Git Needed)
1. Go to the file on GitHub:  
   [`https://github.com/rodrigo-arenas/sklearn-genetic-opt/blob/master/docs-vitepress/versions/latest/community/articles.md`](https://github.com/rodrigo-arenas/sklearn-genetic-opt/blob/master/docs-vitepress/versions/latest/community/articles.md)
2. Click the **Pencil icon (Edit this file)** in the upper right. GitHub will automatically fork the repository for you.
3. Scroll to the bottom of the `## Articles` section.
4. Copy the entry from [`articles_patch.md`](./articles_patch.md), paste it as entry `8.`, and replace the `<YOUR_...>` placeholders with your published URL, name/handle, and platform.
5. Scroll down to **Commit changes**:
   - Commit title: `docs: add community article on search spaces and callbacks`
   - Select **Create a new branch for this commit and start a pull request**.
6. On the PR creation page:
   - Paste the pre-written title and body from [`pull_request_template.md`](./pull_request_template.md).
   - Click **Create pull request**!

#### Method B: Via Git CLI
```bash
# 1. Clone your fork
git clone https://github.com/<YOUR_USERNAME>/sklearn-genetic-opt.git
cd sklearn-genetic-opt

# 2. Create feature branch
git checkout -b docs/add-search-spaces-callbacks-article

# 3. Add the entry to docs-vitepress/versions/latest/community/articles.md
# (Copy from articles_patch.md)

# 4. Commit and push
git add docs-vitepress/versions/latest/community/articles.md
git commit -m "docs: add community article on search spaces and callbacks"
git push origin docs/add-search-spaces-callbacks-article

# 5. Open PR via GitHub using pull_request_template.md
```

---

## 🧪 Validating the Code Locally

To run and verify the tutorial code on your machine:

```bash
cd c:\Users\masfi\OneDrive\Documents\Projects\sklearn-genetic-opt-guide
python tutorial_code.py
```

The script will:
- Load the Wisconsin Breast Cancer diagnostic dataset.
- Define a multi-type search space (`Integer` and `Categorical`).
- Run `GASearchCV` with `ConsecutiveStopping` and `ProgressBar`.
- Demonstrate the early-stopping callback automatically exiting when fitness stabilizes.
- Print a comparative metrics table against `RandomizedSearchCV`.
