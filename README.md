# Cloudscape React + Python API

Full-stack application with React frontend using Cloudscape Design System and FastAPI Python backend.

## Project Structure

```
├── src/                    # React frontend
│   ├── pages/             # React pages
│   ├── services/          # API services
│   └── main.tsx          # App entry point
├── backend/               # Python API
│   ├── main.py           # FastAPI application
│   ├── start.py          # Development server
│   └── requirements.txt  # Python dependencies
└── package.json          # Node.js dependencies
```

## Setup & Development

### Backend (Python API)

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the API server:
```bash
python start.py
```

The API will run on http://0.0.0.0:8000

### Frontend (React)

1. Install Node.js dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will run on http://0.0.0.0:3000

## API Endpoints

- `GET /api/items` - Get all items
- `POST /api/items` - Create new item
- `GET /api/items/{id}` - Get item by ID
- `PUT /api/items/{id}` - Update item
- `DELETE /api/items/{id}` - Delete item
- `GET /health` - Health check

## Production Deployment

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
npm run build
npm run preview
```

## EC2 Configuration

Make sure your EC2 security group allows:
- Port 3000 (React dev server)
- Port 8000 (Python API)
- Port 80/443 (for production)