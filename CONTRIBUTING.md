# Contributing

Hi there 👋

If you're reading this I hope that you're looking forward to adding value to Mayavoz. This document will help you to get started with your journey.

## How to get your code in Mayavoz

1. We use git and GitHub.

2. Fork the mayavoz repository (https://github.com/shahules786/mayavoz) on GitHub under your own account. (This creates a copy of mayavoz under your account, and GitHub knows where it came from, and we typically call this “upstream”.)

3. Clone your own mayavoz repository. git clone https://github.com/ <your-account> /mayavoz (This downloads the git repository to your machine, git knows where it came from, and calls it “origin”.)

4. Create a branch for each specific feature you are developing. git checkout -b your-branch-name

5. Make + commit changes. git add files-you-changed ... git commit -m "Short message about what you did"

6. Push the branch to your GitHub repository. git push origin your-branch-name

7. Navigate to GitHub, and create a pull request from your branch to the upstream repository mayavoz/mayavoz, to the “develop” branch.

8. The Pull Request (PR) appears on the upstream repository. Discuss your contribution there. If you push more changes to your branch on GitHub (on your repository), they are added to the PR.

9. When the reviewer is satisfied that the code improves repository quality, they can merge.

Note that CI tests will be run when you create a PR. If you want to be sure that your code will not fail these tests, we have set up pre-commit hooks that you can install.

**If you're worried about things not being perfect with your code, we will work togethor and make it perfect. So, make your move!**

## Formating

We use [black](https://black.readthedocs.io/en/stable/) and [flake8](https://flake8.pycqa.org/en/latest/) for code formating. Please ensure that you use the same before submitting the PR.


## Testing
We adopt unit testing using [pytest](https://docs.pytest.org/en/latest/contents.html)
Please make sure that adding your new component does not decrease test coverage.

## Other tools
The use of [per-commit](https://pre-commit.com/) is recommended to ensure different requirements such as code formating, etc.

## How to start contributing to Mayavoz?

1. Checkout issues marked as `good first issue`, let us know you're interested in working on some issue by commenting under it.
2. For others, I would suggest you to explore mayavoz. One way to do is to use it to train your own model. This was you might end by finding a new unreported bug or getting an idea to improve Mayavoz.
