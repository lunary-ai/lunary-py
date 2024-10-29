import os
import pytest
from lunary import (
    get_raw_template, 
    get_raw_template_async,
    render_template,
    render_template_async,
    get_langchain_template,
    get_langchain_template_async,
    get_live_templates
)
from lunary.exceptions import TemplateError

slug = "bald-vast-father"
test_data = {"name": "Test User", "message": "Hello World"}

@pytest.fixture(scope="module")
def api_credentials():
    """
    Fixture to load API credentials from environment variables.
    Make sure to set these before running the tests!
    """
    app_id = os.getenv("LUNARY_PRIVATE_KEY", os.getenv("LUNARY_PUBLIC_KEY"))

    api_url = os.getenv("LUNARY_API_URL", "https://api.lunary.ai")  
    
    if not app_id:
        pytest.skip("LUNARY_PUBLIC_KEY environment variable not set")
    
    return {
        "app_id": app_id,
        "api_url": api_url
    }

def test_fetch_existing_template(api_credentials):
    """Test fetching a template that exists"""
    result = get_raw_template(
        slug=slug,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert result is not None
    assert isinstance(result, dict)
    assert "content" in result  

def test_fetch_nonexistent_template(api_credentials):
    """Test fetching a template that doesn't exist"""
    with pytest.raises(TemplateError) as exc_info:
        get_raw_template(
            slug="nonexistent-template-xyz",
            app_id=api_credentials["app_id"],
            api_url=api_credentials["api_url"]
        )
    
    assert "Error fetching template" in str(exc_info.value)

def test_template_caching(api_credentials):
    """Test that template caching works with real API"""
    template1 = get_raw_template(
        slug=slug,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    template2 = get_raw_template(
        slug=slug,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert template1 == template2

# Raw Template Tests (Async)
@pytest.mark.asyncio
async def test_fetch_existing_template_async(api_credentials):
    """Test fetching a template that exists using async function"""
    result = await get_raw_template_async(
        slug=slug,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert result is not None
    assert isinstance(result, dict)
    assert "content" in result  

@pytest.mark.asyncio
async def test_fetch_nonexistent_template_async(api_credentials):
    """Test fetching a template that doesn't exist using async function"""
    with pytest.raises(TemplateError) as exc_info:
        await get_raw_template_async(
            slug="nonexistent-template-xyz",
            app_id=api_credentials["app_id"],
            api_url=api_credentials["api_url"]
        )
    
    assert "Error fetching template" in str(exc_info.value)

@pytest.mark.asyncio
async def test_template_caching_async(api_credentials):
    """Test that template caching works with real API using async function"""
    template1 = await get_raw_template_async(
        slug=slug,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    template2 = await get_raw_template_async(
        slug=slug,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert template1 == template2

def test_render_template_text_mode(api_credentials):
    """Test rendering a template in text mode"""
    result = render_template(
        slug=slug,
        data=test_data,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert result is not None
    assert isinstance(result, dict)
    assert "extra_headers" in result
    assert "Template-Id" in result["extra_headers"]

def test_render_template_messages_mode(api_credentials):
    """Test rendering a template in messages mode"""
    result = render_template(
        slug=slug,
        data=test_data,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    if "messages" in result:
        assert isinstance(result["messages"], list)
        for message in result["messages"]:
            assert "content" in message
            assert "role" in message

@pytest.mark.asyncio
async def test_render_template_text_mode_async(api_credentials):
    """Test rendering a template in text mode using async function"""
    result = await render_template_async(
        slug=slug,
        data=test_data,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert result is not None
    assert isinstance(result, dict)
    assert "extra_headers" in result
    assert "Template-Id" in result["extra_headers"]

@pytest.mark.asyncio
async def test_render_template_messages_mode_async(api_credentials):
    """Test rendering a template in messages mode using async function"""
    result = await render_template_async(
        slug=slug,
        data=test_data,
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    if "messages" in result:
        assert isinstance(result["messages"], list)
        for message in result["messages"]:
            assert "content" in message
            assert "role" in message


def test_get_live_templates(api_credentials):
    """Test fetching live templates"""
    result = get_live_templates(
        app_id=api_credentials["app_id"],
        api_url=api_credentials["api_url"]
    )
    
    assert result is not None
    assert isinstance(result, list)
    if len(result) > 0:
        template = result[0]
        assert "id" in template
        assert "slug" in template