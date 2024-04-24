import asyncio
import os
import pytest
from lunary import (
    get_raw_template,
    get_raw_template_async,
    render_template,
    render_template_async,
    get_langchain_template,
    get_langchain_template_async,
)

@pytest.mark.asyncio
async def test_get_raw_template_async():
    slugs = ["template1", "template2", "template3"]  # Replace with valid template slugs for testing

    tasks = []
    for slug in slugs:
        task = asyncio.create_task(get_raw_template_async(slug))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for result in results:
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "extra" in result

def test_get_raw_template():
    slugs = ["template1", "template2", "template3"]  # Replace with valid template slugs for testing

    for slug in slugs:
        result = get_raw_template(slug)
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "extra" in result

@pytest.mark.asyncio
async def test_render_template_async():
    slugs = ["template1", "template2", "template3"]  # Replace with valid template slugs for testing
    data = {"name": "John", "age": 30}  # Replace with the actual data for rendering

    tasks = []
    for slug in slugs:
        task = asyncio.create_task(render_template_async(slug, data))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for result in results:
        assert isinstance(result, dict)
        assert "text" in result or "messages" in result
        assert "extra_headers" in result

def test_render_template():
    slug = "template1" 
    data = [{"name": "John", "age": 30},{"name": "Jane", "age": 18}]

    for datum in data:
        result = render_template(slug, datum)
        assert isinstance(result, dict)
        assert "text" in result or "messages" in result
        assert "extra_headers" in result

@pytest.mark.asyncio
async def test_get_langchain_template_async():
    slugs = ["template1", "template2", "template3"]  

    tasks = []
    for slug in slugs:
        task = asyncio.create_task(get_langchain_template_async(slug))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for result in results:
        assert result is not None

def test_get_langchain_template():
    slugs = ["template1", "template2", "template3"]  

    for slug in slugs:
        result = get_langchain_template(slug)
        assert result is not None