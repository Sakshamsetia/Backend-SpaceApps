# ----------------------------------------------------------------------
# STAGE 1: BUILDER (Build environment for installing packages)
# ----------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Copy dependencies from the root of the GitHub repo (where Dockerfile is)
COPY requirements.txt . 
# Install packages. This includes Gunicorn, which MUST be in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# ----------------------------------------------------------------------
# STAGE 2: PRODUCTION RUNTIME (Minimal image for execution)
# ----------------------------------------------------------------------
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the environment and installed packages from the builder stage
COPY --from=builder /app /app

# Security Best Practice: Create a non-root user
RUN adduser --disabled-password myuser
USER myuser

# Set the default port for Cloud Run
ENV PORT 8080

# The Gunicorn command to start the Django application.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "my_django_app.src.my_django_app.wsgi:application"]
