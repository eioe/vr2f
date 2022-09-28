# Git Workflow

## General points
* regularly `git pull` the `main` branch
* minimize the usage of branching to avoid conflicts
* if `branching` is necessary, try to `merge` into `main` whenever possible
* when working on your own script you can work directly in the `main`, since you can assume that only you work on the script
* When working on a script of another main author or at the code base, which might have global effects, we work with a `branch` (see below)
    * Best practice is to communicate this step in advance via any channel.
* `rebase` only if you want to keep the `commit` history of a feature `branch` (see below)

## Scenarios

### Creating and solving `issues` on *GitLab*
`Issues` can be used for project management and general overviews.

Each `issue` gets a reference number. When working on an issue, `commit` messages can refer to an `issue` by including its number, e.g.:

```shell
# change something in a script according to an open `issue`
# then commit the changes 
git commit SCRIPT.py -m "#3 I did this & that."  # where '#3' refers to the `issue` number
```

This [video](https://www.youtube.com/watch?v=jhtbhSpV5YA) describes this workflow (here this is done on *GitHub*, but for *GitLab* it works the same way).

### Working with a `branch`

#### What is a `branch`
A `branch` creates another line of development in the project and enables you to track each version of your project systematically. The default `branch` in your project is the  `main` (or `master` @*GitHub*) `branch`.

Branches are independent of each other. Changes made in one `branch` do not affect all other branches (until they are `merged` back together)

#### How to create, work in & delete a `branch`

##### Create a `branch`

```shell
git checkout -b [name_of_new_branch]
```
**! Important !** : Before creating a new `branch`, `pull` changes from the project. Your `main` (`master`) needs to be up-to-date. 

##### Work with them 
While a `branch` is active, make changes in the code.
After the work ist done `commit`the changes in the usual way. 
To `push` the changes to the `remote` repository, do: 

```shell
git push -u origin [name_of_new_branch]  # this pushes the branch with its changes into GitLab/GitHub
```
A single `git push` is enough, if the `branch` is already in the `remote`. 

Switching to another `branch`
```shell
git checkout [name_of_existing_branch]  # e.g., main
```

List all `branches` (local and remote)
```shell
git branch -a
```

##### Delete them
Delete `branch` locally
```shell
git branch -d [name_of_local_branch]
```

Delete `branch` remotely
```shell
git push origin --delete [name_of_remote_branch]
```
A `branch` can be deleted also in the web surface of the repository. This is usually done after a certain project feature is implemented and `merged` into the `main`. 

#### How to `merge` the `branch` into the `main`
Merging can be done in the web interface, and comes with a lot of interactive elements (e.g, comment functions) [prefered option].

Via the shell it can be done this way:

```shell
git checkout [name_of_branch]
# Do something and commit 
git checkout main
git pull origin main  # update the local master/main
git merge [name_of_branch]  # merge locally
git push origin main. # push into remote main
```

#### How to `rebase` the `branch` into the `main`
`Rebasing` takes all `commits`from a feature `branch` and puts them on top of `main`/`master` branch. 
In contrast, `merge` pools all `commits`of the feature `branch` and puts them in one big `merge` `commit`. 
Thus, `rebasing` has the advantage to keep the history of the changes, also those which were done in a feature `branch`. 

How it is done: 

```shell
git checkout [name_of_branch]  # create new local branch
# Do something and commit 
# ...
git checkout main  # switch to main/master
 git pull  # update the local main/master 
git checkout [name_of_branch]  # switch to branch  
# merge locally
git rebase main  # rebasing the feature branch on top the master, i.e. it moves the commit history up the stream 
# potentially solve conflicts
# ...
git checkout main  # switch to main/master
git rebase [name_of_branch]  # this now takes the commits of the feature branch and stacks them on top of the commit history of the main/master branch
git push  # push to remote
```
This `rebase` workflow is based on [blogpost + video](https://www.themoderncoder.com/a-better-git-workflow-with-rebase/).
`rebase` comes with some caveats (referenced in the blog post). So mainly use it, when it is important to keep the `commit`-history of a feature-`branch`.

### How to solve merge conflicts
If there are conflicts between different versions of a script, `git` will ask you to solve this.
Open the respective script and search for following format:

```
<<<<<<< HEAD
Code/Edits as they are in the main branch
=======
Code/Edits as they are done by you.
>>>>>>> new_branch_to_merge_later
```
It is up to you how to `merge` the two versions within this format. 
What needs to be done after `merging` is to remove the conflict skeleton:  
```
# REMOVE this from code after merging two versions. 
<<<<<<< HEAD

=======

>>>>>>> new_branch_to_merge_later
```

Then just `commit` changes and `push`

```shell
git commit FILE_WITH_CONFLICT.py -m "resolved the merge conflict"
git push
```

## Additional references
* [Book chapter on rebasing](http://git-scm.com/book/en/v2/Git-Branching-Rebasing)
