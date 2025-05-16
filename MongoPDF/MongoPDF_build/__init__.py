from mcp.server.fastmcp import FastMCP, Context
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import asyncio
import logging
import sys
from bson import ObjectId
import os
import io
from PyPDF2 import PdfReader
from datetime import datetime
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("mongodb-mcp")

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app):
    """MongoDB connection lifecycle management"""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set")
    
    client = None
    try:
        client = AsyncIOMotorClient(uri)
        await client.admin.command('ping')
        logger.info("MongoDB connection established")
        yield {"client": client}
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        if client:
            client.close()
        raise
    finally:
        if client:
            client.close()

# Create MCP server instance
mcp = FastMCP("MongoDB MCP Server", lifespan=lifespan)


def categorize_file(file_data: Dict[str, Any]) -> List[str]:
    """Automatically classify file based on its metadata and filename."""
    tags: List[str] = []
    meta = file_data.get("metadata", {})
    if meta.get("Application-Oriented", False):
        tags.append("Application")
    if file_data.get("filename", "").lower().endswith(".pdf"):
        tags.append("Document")
    if not tags:
        tags.append("General")
    return tags


@mcp.tool()
async def ping(ctx: Context) -> Dict[str, Any]:
    """Test MongoDB connection"""
    try:
        client = ctx.request_context.lifespan_context["client"]
        await client.admin.command('ping')
        return {"status": "ok", "message": "Connected to MongoDB"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
async def list_databases(ctx: Context) -> Dict[str, Any]:
    """List all available databases"""
    try:
        client = ctx.request_context.lifespan_context["client"]
        databases = await client.list_database_names()
        return {"status": "ok", "databases": databases}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
async def find_documents(
    database: str,
    collection: str,
    ctx: Context,
    query: Dict[str, Any] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Find documents in a collection"""
    try:
        client = ctx.request_context.lifespan_context["client"]
        db = client[database]
        coll = db[collection]
        cursor = coll.find(query or {}).limit(limit)
        documents = await cursor.to_list(length=limit)
        return {"status": "ok", "documents": documents}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def classify_and_write_metadata(ctx: Context, file_id: str) -> Dict[str, Any]:
    """
    Classify a file by metadata and write tags back to MongoDB.
    Returns status and list of tags.
    """
    try:
        client = ctx.request_context.lifespan_context["client"]
        db = client[os.getenv("MONGODB_DB")]
        files_coll = db["fs.files"]
        file_data = await files_coll.find_one({"_id": ObjectId(file_id)})
        if not file_data:
            return {"status": "error", "message": "File not found"}
        tags = categorize_file(file_data)
        await files_coll.update_one(
            {"_id": ObjectId(file_id)}, {"$set": {"tags": tags}}
        )
        return {"status": "ok", "message": f"File {file_id} classified", "tags": tags}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def list_metadata_tags(ctx: Context) -> Dict[str, Any]:
    """
    List all distinct metadata tags for PDF files.
    """
    client = ctx.request_context.lifespan_context["client"]
    db = client[os.getenv("MONGODB_DB")]
    tags = await db["fs.files"].distinct("tags")
    return {"status": "ok", "tags": tags}


@mcp.tool()
async def find_pdfs_by_tag(ctx: Context, tag: str) -> Dict[str, Any]:
    """
    Find PDF files matching a given metadata tag.
    Returns file IDs and filenames.
    """
    client = ctx.request_context.lifespan_context["client"]
    db = client[os.getenv("MONGODB_DB")]
    cursor = db["fs.files"].find({"tags": tag})
    files = []
    async for doc in cursor:
        files.append({"file_id": str(doc["_id"]), "filename": doc["filename"]})
    return {"status": "ok", "files": files}


@mcp.tool()
async def fetch_pdf_fulltext(ctx: Context, file_id: str) -> Dict[str, Any]:
    """
    Retrieve full text of a PDF stored in GridFS and return it. It location is fs.files
    """
    client = ctx.request_context.lifespan_context["client"]
    db = client[os.getenv("MONGODB_DB")]
    file_doc = await db["fs.files"].find_one({"_id": ObjectId(file_id)})
    if not file_doc:
        return {"status": "error", "message": "File not found"}
    stream = io.BytesIO()
    async for chunk in (
        db["fs.chunks"].find({"files_id": ObjectId(file_id)}).sort("n", 1)
    ):
        stream.write(chunk["data"])
    stream.seek(0)
    reader = PdfReader(stream)
    text_pages = [p.extract_text() or "" for p in reader.pages]
    full_text = "\n\n".join(text_pages)
    return {"status": "ok", "text": full_text}

if __name__ == "__main__":
    logger.info("Starting MongoDB MCP Server...")
    mcp.run()
