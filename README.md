# Natural Language Processing
Repository for the NLP class of the Big Data and Analytics Master. 

In this repository you will find all the technical materials related to this course (notebooks, scripts, files,...).

Please read carefully the following instructions to setup the environment needed for the practices

# Using the NLP environment

This section covers the instructions to configure the repo for executing locally the practice notebooks. If you do not have a GPU or you prefer to avoid the complexity of the local configuration and to execute them remotely, please refer to the section [Working with GPUs](#working-with-gpus)

## Create the working directory

On your computer, create the directory on which you want to work. In my case, it's called `NLP-MBD-EN2021S-Electives`.

## Connect your local directory to this repository

In this step you will connect your local directory to this repository. In this way you will download the data initially included here, as well as the new materials that I will be uploading regularly.

The first thing you need to do is to open a terminal and go to the folder that you previously created (please note that the following is where I have the directory, change the path with yours)

```
cd /
```  
Clone the repository
```
git clone https://github.com/acastellanos-ie/NLP-MBD-EN2022A-Electives.git
```

After cloning you should see all the materials in your local folder (and your local folder will be connected to this repo).

I will be regularly updating the materials, so, please, before each practical class make sure that you have the latest version of the repo ([Guide](https://www.atlassian.com/git/tutorials/syncing/git-pull). for pulling data from a repo)

## Download Anaconda

I strongly recommend you to use Anaconda to manage your Python environments. [Link to download Anaconda](https://www.anaconda.com/products/individual)

## Create the `conda` environment

`conda` environments are the playgrounds on which we’ll work in Python. There are other tools to create environments too, but `conda` is included and installed when installing the
Anaconda suite and is one the industry’s standards.

1. Open a terminal and go the folder you previously created

  ```
  cd /
  ```
  
2. Create the environment with a name of your choice:

  ```
  conda create --name nlp python=3.8
  ```
  
After the parameter `--name` we define the name of our environment. In my case, it’s `nlp`. Also you can choose the python version (3.8 for us).

3. Activate your environment:

  ```
  conda activate nlp
  ```
  
## Install the required libraries in your environment

In order to have a smooth and stable environment, it’s common to have a `requirements.txt` file that includes all the libraries that are used in your projects. 

That way, whenever you join a new project, it’s easy to start working without worrying about missing libraries or modules.

Installing the required libraries in our environment it’s as easy as follows (the requirements.txt is the file included in this repository):

```
pip install -Uqqr requirements.txt
```

## Open a Jupyter/JupyterLab session with our current environment

Before opening a Jupyter session, we have to make sure that our environments will be included. In order to solve this, run the following command:

```
conda install nb_conda_kernels
```

Once this is installed, we are ready to open a Jupyter session by using the following command:

```
jupyter lab
```

or if you prefer the old notebooks use:

```
jupyter notebook
```

A new tab of your browser will open with the Jupyter session, showing all the available environments. To create a Notebook with your current environment, select it from the
Notebooks list.

And now you’re ready to start!

# Working with GPUs

As we will cover in class, the most advanced NLP methodologies nowadays are based on Deep Learning models. These models, while providing an impressive performance in many task, have the limitation of their complexity and the computation power they require.

In particular, in order to use these models, it is highly recommended to have a GPU at your disposal. As I imagine that most of you do not have one in your compute, my recommendation is to leverage [Google Colab](https://colab.research.google.com/).

Google Colab provides you an environment in which you can execute Python code and have access to GPUs. If you want to execute any of the notebooks in this repository in Colab, you can just click on the Open in Colab button at the beggining of each notebook
