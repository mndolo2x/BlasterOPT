# 🧹 Google Cloud Platform (GCP) Manual Resource Cleanup Guide

This document provides exact step-by-step `gcloud` CLI commands to manually inspect and delete legacy Google Cloud resources created during earlier deployment testing (Vertex AI endpoints, Cloud Run services, GKE clusters, Compute Engine VMs, and Cloud Storage buckets).

> ⚠️ **Note:** Do NOT run commands automatically. Verify your active GCP project ID before executing any resource deletion.

---

## 🛠️ Step 0: Set Active GCP Project & Authenticate

Open your local terminal or Google Cloud Shell and authenticate:

```bash
# 1. Authenticate with Google Cloud
gcloud auth login

# 2. Set your active Google Cloud Project ID
export GCP_PROJECT_ID="YOUR_PROJECT_ID_HERE"
gcloud config set project $GCP_PROJECT_ID
```

---

## 1. 🤖 Vertex AI Model Garden & Endpoints Cleanup

List and undeploy Vertex AI endpoints:

```bash
# List all Vertex AI endpoints in region us-central1
gcloud ai endpoints list --region=us-central1

# Undeploy models from the endpoint
# Usage: gcloud ai endpoints undeploy-model ENDPOINT_ID --deployed-model-id=DEPLOYED_MODEL_ID --region=us-central1
gcloud ai endpoints list --region=us-central1 --format="value(name)"

# Delete Vertex AI endpoint
# Replace ENDPOINT_ID with your actual endpoint ID
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1 --quiet
```

---

## 2. 🚀 Cloud Run Services Cleanup

List and delete Cloud Run services:

```bash
# List Cloud Run services
gcloud run services list --region=us-central1

# Delete Cloud Run service
gcloud run services delete blasteropt-app --region=us-central1 --quiet
```

---

## 3. ☸️ Google Kubernetes Engine (GKE) Clusters Cleanup

List and delete GKE clusters:

```bash
# List GKE clusters
gcloud container clusters list --region=us-central1

# Delete GKE cluster
gcloud container clusters delete blasteropt-cluster --region=us-central1 --quiet
```

---

## 4. 💻 Compute Engine Virtual Machines (VMs) Cleanup

List and delete Compute Engine instances and GPU reservations:

```bash
# List Compute Engine VM instances
gcloud compute instances list

# Delete Compute Engine VM instance
gcloud compute instances delete blasteropt-vm --zone=us-central1-a --quiet
```

---

## 5. 🪣 Cloud Storage (GCS) Buckets Cleanup

List and delete Cloud Storage buckets and container artifacts:

```bash
# List Cloud Storage buckets
gcloud storage buckets list

# Delete Cloud Storage bucket and all contents
# Usage: gcloud storage rm --recursive gs://BUCKET_NAME
gcloud storage rm --recursive gs://blasteropt-artifacts-$GCP_PROJECT_ID
```

---

## 🐳 Artifact Registry & Docker Images Cleanup

```bash
# List Artifact Registry repositories
gcloud artifacts repositories list --location=us-central1

# Delete Artifact Registry repository
gcloud artifacts repositories delete blasteropt-repo --location=us-central1 --quiet
```

---

## ✅ Verification Checklist

Confirm all resources have been successfully released to avoid recurring billing:

```bash
gcloud ai endpoints list --region=us-central1
gcloud run services list
gcloud container clusters list
gcloud compute instances list
```
