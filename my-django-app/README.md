# My Django App

This is a Django application running with Gunicorn as a WSGI server.

## Project Structure

```
my-django-app
├── src
│   ├── manage.py
│   ├── my_django_app
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   └── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Getting Started

To get started with this project, you need to have Docker installed on your machine.

### Building the Docker Image

Run the following command to build the Docker image:

```
docker build -t my-django-app .
```

### Running the Application

After building the image, you can run the application using:

```
docker run -p 8080:8080 my-django-app
```

The application will be accessible at `http://localhost:8080`.

## Dependencies

The project dependencies are listed in `src/requirements.txt`. Make sure to update this file as needed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.