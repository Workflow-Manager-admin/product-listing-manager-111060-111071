from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
import io
import csv
import os

import psycopg2
from psycopg2.extras import RealDictCursor

# --- ENV & DB Connection ---
from dotenv import load_dotenv
load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def get_db_connection():
    """Utility to get a database connection."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_URL,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            cursor_factory=RealDictCursor,
        )
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

app = FastAPI(
    title="Product Listing Manager API",
    description="API for e-commerce multi-channel product listing manager backend (excels, images, products, channel integration, sync, analytics stub).",
    version="0.1.0",
    openapi_tags=[
        {"name": "products", "description": "Product CRUD, listing, resync"},
        {"name": "files", "description": "Excel and image upload & preview"},
        {"name": "channels", "description": "Channel & country selection, integrations"},
        {"name": "analysis", "description": "Image Analysis Stub"},
        {"name": "status", "description": "Real-time/polling status endpoints"},
    ]
)

# Allow CORS for dev/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models & Schemas ---
class ProductBase(BaseModel):
    name: str = Field(..., description="Product name")
    description: Optional[str] = Field(None, description="Description")
    price: float = Field(..., description="Price")
    stock: int = Field(..., description="Stock quantity")
    sku: Optional[str] = Field(None, description="SKU/identifier")
    main_image_url: Optional[str] = Field(None, description="Primary image url")
    # Add more core attributes as needed

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str]
    description: Optional[str]
    price: Optional[float]
    stock: Optional[int]
    main_image_url: Optional[str]
    sku: Optional[str]

class ProductResponse(ProductBase):
    id: int
    channels: List[str] = []
    countries: List[str] = []
    # Add more fields (sync status, listing IDs, etc)

class ChannelCountrySelection(BaseModel):
    product_id: int = Field(..., description="Product ID")
    channels: List[str] = Field(..., description="Channels to sync (ex: ['Lazada', 'Shopee'])")
    countries: List[str] = Field(..., description="Target countries for listing (ex: ['SG', 'MY'])")

class ListingStatusResponse(BaseModel):
    id: int
    sync_status: str
    last_updated: Optional[str]
    channel_statuses: Dict[str, Any] = Field({}, description="Status per channel")

class ImageAnalysisOutput(BaseModel):
    product_name: str
    attributes: Dict[str, Any] = Field({}, description="Auto-filled attributes")
    # Can add more prefill details

# --- Utility Functions ---
def product_row_to_response(row):
    # Helper to map DB row to ProductResponse
    out = dict(row)
    # channels, countries are stored as arrays/strings in DB, convert if needed
    if 'channels' in out and isinstance(out['channels'], str):
        out['channels'] = out['channels'].split(",")
    if 'countries' in out and isinstance(out['countries'], str):
        out['countries'] = out['countries'].split(",")
    return ProductResponse(**out)

# --- Endpoints ---

# PUBLIC_INTERFACE
@app.get("/", tags=["status"])
def health_check():
    """Health check endpoint."""
    return {"message": "Healthy"}

# PRODUCT CRUD

# PUBLIC_INTERFACE
@app.post("/products/", response_model=ProductResponse, tags=["products"], summary="Create Product", description="Create a new product.")
def create_product(payload: ProductCreate):
    """Create a new product."""
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO products (name, description, price, stock, sku, main_image_url)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING *;
        """, [payload.name, payload.description, payload.price, payload.stock, payload.sku, payload.main_image_url])
        row = cur.fetchone()
    conn.close()
    return product_row_to_response(row)

# PUBLIC_INTERFACE
@app.get("/products/", response_model=List[ProductResponse], tags=["products"], summary="List Products", description="List all products.")
def list_products(
    skip: int = 0, 
    limit: int = 50
):
    """List all products, paginated."""
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM products ORDER BY id DESC OFFSET %s LIMIT %s;", [skip, limit])
        rows = cur.fetchall()
    conn.close()
    return [product_row_to_response(row) for row in rows]

# PUBLIC_INTERFACE
@app.get("/products/{product_id}", response_model=ProductResponse, tags=["products"], summary="Get Product", description="Get a product by ID.")
def get_product(product_id: int):
    """Retrieve product by ID."""
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM products WHERE id=%s;", [product_id])
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Product not found")
    conn.close()
    return product_row_to_response(row)

# PUBLIC_INTERFACE
@app.put("/products/{product_id}", response_model=ProductResponse, tags=["products"], summary="Update Product", description="Update a product by ID.")
def update_product(product_id: int, payload: ProductUpdate):
    """Update product."""
    v = payload.dict(exclude_unset=True)
    if not v:
        raise HTTPException(status_code=400, detail="No fields to update")
    fields = [f"{k} = %s" for k in v]
    values = list(v.values())
    values.append(product_id)
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute(f"UPDATE products SET {', '.join(fields)} WHERE id = %s RETURNING *;", values)
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Product not found")
    conn.close()
    return product_row_to_response(row)

# PUBLIC_INTERFACE
@app.delete("/products/{product_id}", response_model=dict, tags=["products"], summary="Delete Product", description="Delete product by ID.")
def delete_product(product_id: int):
    """Delete a product."""
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM products WHERE id = %s RETURNING id;", [product_id])
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Product not found")
    conn.close()
    return {"deleted": product_id}

# Endpoint for resync/edit listing
# PUBLIC_INTERFACE
@app.post("/products/{product_id}/resync", response_model=ListingStatusResponse, tags=["products"], summary="Resync Product Listings", description="Trigger re-sync of product listings to all assigned channels.")
def resync_product_listing(product_id: int):
    """Trigger re-sync for a product's channel listings (mock implementation)."""
    # In actual, would kick off background sync task here
    return ListingStatusResponse(
        id=product_id,
        sync_status="queued",
        last_updated=None,
        channel_statuses={}
    )

# --- Excel/CSV Upload & Parsing ---

# PUBLIC_INTERFACE
@app.post("/upload/excel", tags=["files"], summary="Upload Excel", description="Upload an Excel/CSV file and parse as products.")
async def upload_excel(file: UploadFile = File(...)):
    """Upload and parse Excel (CSV for demo; extend to .xlsx if needed). Returns parsed contents."""
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="File must be .csv or .xlsx")
    # Simple handling for CSV (for demo)
    content = await file.read()
    content_io = io.StringIO(content.decode("utf-8"))
    reader = csv.DictReader(content_io)
    products = [row for row in reader]
    return {"parsed_products": products}

# --- Image Upload/Preview Endpoint ---

@app.post("/upload/image", tags=["files"], summary="Upload Product Image", description="Upload an image file for a product. Returns preview URL (simulation).")
async def upload_image(file: UploadFile = File(...)):
    """Upload product image and get preview URL."""
    # In production, save file securely, e.g., object storage
    save_dir = "./uploaded_images"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    # Pretend preview URL (normally would be static serving endpoint or S3 etc)
    preview_url = f"/preview/image/{file.filename}"
    return {"image_url": preview_url}

@app.get("/preview/image/{file_name}", tags=["files"], summary="Preview Uploaded Image", description="Returns an uploaded image")
def preview_image(file_name: str):
    """Serve uploaded images for preview."""
    save_dir = "./uploaded_images"
    file_path = os.path.join(save_dir, file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    def iterfile():
        with open(file_path, "rb") as f:
            yield from f
    return StreamingResponse(iterfile(), media_type="image/jpeg")

# --- Image Analysis / Prefill Stub ---

@app.post("/analysis/image", response_model=ImageAnalysisOutput, tags=["analysis"], summary="Analyze Product Image for Prefill", description="Analyzes a product image and returns pre-filled attributes (mock implementation).")
async def analyze_image(file: UploadFile = File(...)):
    """Image analysis stub - in actual app, call ML model/API to pre-fill attributes."""
    # For now, return a mock; typically, you would use: await file.read()
    # ...actual analysis code here...
    # Pretend it found a product name and some attributes:
    return ImageAnalysisOutput(
        product_name="Auto-detected Example Product",
        attributes={"color": "blue", "category": "shoes"}
    )

# --- Channel + Country Selection & Save ---

@app.post("/select_channels", tags=["channels"], summary="Select Channels & Countries", description="Assign a product to one or more channels and countries.")
def select_channels(payload: ChannelCountrySelection):
    """Assign a product to channels/countries (stores associations in DB)."""
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        # Insert/update join table/fields as per schema
        cur.execute("""
            UPDATE products
            SET channels=%s, countries=%s
            WHERE id=%s
            RETURNING *;
        """, [",".join(payload.channels), ",".join(payload.countries), payload.product_id])
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Product not found")
    conn.close()
    return product_row_to_response(row)

# --- Channel Listing/Integration Modules (Stubs) ---

class ChannelBase:
    """Base class for all channel integrations. Extendable for real APIs."""
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        # Load other config as needed

    def list_product(self, product: dict):
        """Stub for sending product to channel API."""
        raise NotImplementedError("Implement in subclass")

    def update_product(self, product_id: str, fields: dict):
        raise NotImplementedError("Implement in subclass")

class LazadaChannel(ChannelBase):
    def list_product(self, product: dict):
        # Place logic here to integrate with Lazada's API
        return {"result": "Listed to Lazada (stub)"}

class ShopeeChannel(ChannelBase):
    def list_product(self, product: dict):
        return {"result": "Listed to Shopee (stub)"}

class TikTokChannel(ChannelBase):
    def list_product(self, product: dict):
        return {"result": "Listed to TikTok (stub)"}

# Example endpoint for modular integration stubs
@app.post("/channels/{channel}/list_product", tags=["channels"], summary="List product to channel", description="(Stub) List a product to a connected channel.")
def list_to_channel(channel: str, product_id: int):
    """Stub endpoint - forwards to modular integration stub."""
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM products WHERE id=%s;", [product_id])
        row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")
    product_dict = dict(row)
    result = None
    if channel.lower() == "lazada":
        # Place logic to handle real API keys from .env here (LAZADA_API_KEY, etc)
        result = LazadaChannel().list_product(product_dict)
    elif channel.lower() == "shopee":
        result = ShopeeChannel().list_product(product_dict)
    elif channel.lower() == "tiktok":
        result = TikTokChannel().list_product(product_dict)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown channel {channel}")
    return {"channel": channel, "result": result}

# --- Real-time status endpoints (pollable) ---

@app.get("/status/listing/{product_id}", tags=["status"], response_model=ListingStatusResponse, summary="Poll Listing Status", description="Get the synchronization/listing status for a product.")
def poll_listing_status(product_id: int):
    # In a real app, would check per-channel listing/job/task status
    return ListingStatusResponse(
        id=product_id,
        sync_status="in_progress",
        last_updated=None,
        channel_statuses={}
    )

# --- Misc: Channel/country info endpoints, for UI dropdowns ---
@app.get("/channels/available", tags=["channels"], summary="Available Channels", description="List supported channels.")
def available_channels():
    """Channels supported by the platform."""
    return {"channels": ["Lazada", "Shopee", "TikTok"]}

@app.get("/countries/available", tags=["channels"], summary="Available Countries", description="List supported countries.")
def available_countries():
    # Could make this dynamic/configurable
    return {"countries": ["SG", "PH", "MY", "TH", "VN", "ID"]}

# --- API Docs Note ---
@app.get("/docs/websocket_note", tags=["status"], summary="API Usage Note", description="WebSocket note (not used). Only REST polling is supported for status updates at this time.")
def websocket_note():
    return {"note": "This API does NOT use websockets for real-time updates. Use the polling endpoints under /status."}

# -- ENDPOINT for OpenAPI regeneration (for dev usage only)
@app.post("/generate_openapi", tags=["status"], summary="Regenerate OpenAPI", description="Regenerate OpenAPI spec file in interfaces/openapi.json")
def regen_openapi():
    openapi_schema = app.openapi()
    out_path = os.path.join(os.path.dirname(__file__), '../../interfaces/openapi.json')
    with open(out_path, "w") as f:
        import json
        json.dump(openapi_schema, f, indent=2)
    return {"status": "OK"}

# --- INSTRUCTIONS FOR API KEYS ---
# Place channel API keys/config in .env file using names like:
#   LAZADA_API_KEY=...
#   SHOPEE_API_KEY=...
#   TIKTOK_API_KEY=...
# See comments in ChannelBase subclasses for usage.
