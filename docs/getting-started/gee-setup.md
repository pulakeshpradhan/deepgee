# GEE Setup

Detailed guide for setting up Google Earth Engine.

## Overview

Google Earth Engine (GEE) is a cloud-based platform for planetary-scale geospatial analysis. DeepGEE uses GEE to access and process satellite imagery.

## Prerequisites

- Google account
- Internet connection
- Web browser

## Step-by-Step Setup

### 1. Sign Up for Google Earth Engine

1. Visit [https://earthengine.google.com/](https://earthengine.google.com/)
2. Click **"Sign Up"** or **"Get Started"**
3. Sign in with your Google account
4. Fill out the registration form:
   - Select your use case (Research, Education, etc.)
   - Provide your affiliation
   - Describe your intended use
5. Submit the form
6. Wait for approval email (usually 24-48 hours)

!!! info "Approval Time"
    GEE approval typically takes 1-2 days. You'll receive an email when approved.

### 2. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Select a project"** → **"New Project"**
3. Enter project details:
   - **Project name**: e.g., "deepgee-project"
   - **Organization**: (optional)
   - **Location**: (optional)
4. Click **"Create"**
5. Note your **Project ID** (e.g., `deepgee-project-123456`)

### 3. Enable Earth Engine API

1. In Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for "Earth Engine API"
3. Click on "Earth Engine API"
4. Click **"Enable"**

### 4. Set Up Authentication

#### Method 1: Notebook Authentication (Recommended)

```python
import deepgee

# This opens a browser for authentication
deepgee.authenticate_gee(auth_mode='notebook')
```

#### Method 2: gcloud Authentication

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project your-project-id
```

Then in Python:

```python
import deepgee
deepgee.initialize_gee(project='your-project-id', auth_mode='gcloud')
```

#### Method 3: Service Account

For automated workflows:

1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Use in Python:

```python
import deepgee
deepgee.initialize_gee(
    project='your-project-id',
    service_account='your-service-account@project.iam.gserviceaccount.com',
    key_file='path/to/key.json'
)
```

## Verify Setup

Check that everything is working:

```python
import deepgee

# Initialize
deepgee.initialize_gee(project='your-project-id')

# Check status
status = deepgee.auth.check_gee_status()
print(status)

# Expected output:
# {
#     'status': 'authenticated',
#     'project': 'your-project-id',
#     'user': 'your-email@gmail.com'
# }
```

## Troubleshooting

### Authentication Failed

```python
# Re-authenticate
deepgee.authenticate_gee()

# Check credentials
import ee
print(ee.data.getAssetRoots())
```

### Project Not Found

Make sure you're using the **Project ID**, not the project name.

### Permission Denied

Ensure the Earth Engine API is enabled for your project.

## Best Practices

!!! tip "Project Organization"
    Create separate projects for different applications to manage quotas and billing.

!!! tip "Service Accounts"
    Use service accounts for production deployments and automated workflows.

!!! warning "Quotas"
    Be aware of GEE usage quotas. Monitor your usage in the Google Cloud Console.

## Next Steps

- [Quick Start](quick-start.md) - Start using DeepGEE
- [Data Download Guide](../user-guide/data-download.md) - Learn about data access
- [Examples](../examples/land-cover.md) - See complete workflows
