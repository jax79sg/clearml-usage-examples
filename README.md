This set of articles serves to guide you through the steps of model development with the AI Tech Stack. We hope that you could quickly look through the overview before proceeding with the detailed chapters.

# Overview
A ML model is almost always developed as part of an application. As such, it is necessary to ensure that the model developed be maintained to sustain the application in mind. The following diagram shows the overall process of a typical ML model development cycle and how it serves its application. In our organisation, we use the term ML Ops to describe this process.

PICTURE OF ML OPS

Pretty much all ML development starts with Data gathering, and some Exploratory **Data Analysis**. In the process, we have a better understanding of the data and ML tasks and starts working on **Feature Engineering** and **Model Development**. During this period, we would find ourselves manipulating the data a lot so that it can fit into our model. Over time we may lose track of what changes the data has gone through and find ourselves unable to reproduce results, this is where **Data Versioning** will help us. While we develop a model for a task, we will subject our models to multiple training rounds with different hyperparameters and model architectures (This we will term as Experiments from now on). Above all will have some kind of codes, so **Code Versioning** will be required as well. Very frequently we will want to look back at the results and reference them to the hyperparams or network architectures we used in the model. To do this, some form of **Experiment Tracking** is required. Models trained will also need to be tracked so we know how the model was achieved and how we can share the models to others, this we call **Model Versioning**. When we finalise on a model, we need to ensure the model is robust enough to actually be used in production or prototype environments where real data is expected. A series of **Model Testing** should be perform to ensure this. Remember that our models are always meant to serve an application, so its a no brainer to make the model available (**Model Serving**) to the main application and to test the models behaviours (Both normal and abnormal) against the application. This process is what we term as **Contextualised Testing**. We finally deploy the model in production but as all good ML Practioners knows, no ML model will retain its performance in production forever without intervention. So some form of monitoring in terms of **Model Performance** and **Data Drifts** needs to be engaged to ensure the the model doesn't stale out without notice, but to monitor we need some kind of ground truths so production **Data Labelling** is unavoidable as well. When performance issues occurs, a **Model Retrain** or even **Model Redevelopment** is due.   

# Infra Available
**ECS** - [Using S3 Server](s3server) <br>

# Tools Available
**Imagery Data Labelling** - CVAT<br>
**Data Analysis, Feature Engineering** - Dataiku<br>
**Data Versioning** - ClearML and S3<br>
**Model Versioning** - ClearML and S3<br>
**Code Versioning** - GitLab<br>
**Experiment Tracking** - ClearML<br>
Check out [How to do Experiment Tracking on ClearML](clearml) for detailed write up and examples.<br>
**Model Serving** - NVidia Triton Server<br>
**Data Drift Detection** - Alibi Detection<br>
**Automated Model Retrain** - ClearML and Jenkins/Gitlab Runner<br>

# More Research required
Model Testing<br>
Model Performance<br>
Data Drift<br>
Contextualised Testing<br>

