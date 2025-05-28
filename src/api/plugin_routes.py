"""
API routes for plugin management.

This module provides REST endpoints for discovering, loading, and using plugins.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel
import structlog

from ..plugins import PluginType, PLUGIN_REGISTRY, PLUGIN_LOADER
from ..core.models import Action, Principle
from .middleware import get_current_user

logger = structlog.get_logger()

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


class PluginListResponse(BaseModel):
    """Response model for plugin listing."""
    plugins: Dict[str, List[str]]
    total_count: int


class PluginMetadataResponse(BaseModel):
    """Response model for plugin metadata."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    dependencies: List[str]
    config_schema: Optional[Dict[str, Any]]
    tags: List[str]


class LoadPluginRequest(BaseModel):
    """Request model for loading a plugin."""
    plugin_name: str
    plugin_type: str
    config: Optional[Dict[str, Any]] = None


class UsePluginRequest(BaseModel):
    """Request model for using a plugin."""
    plugin_name: str
    plugin_type: str
    method: str
    data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


@router.get("/list", response_model=PluginListResponse)
async def list_plugins(
    plugin_type: Optional[str] = None,
    user_id: str = Depends(get_current_user)
) -> PluginListResponse:
    """
    List all available plugins.
    
    Args:
        plugin_type: Optional filter by plugin type
        
    Returns:
        List of available plugins grouped by type
    """
    try:
        # Convert string to PluginType if provided
        filter_type = None
        if plugin_type:
            filter_type = PluginType(plugin_type)
        
        plugins = PLUGIN_REGISTRY.list_plugins(filter_type)
        total_count = sum(len(p) for p in plugins.values())
        
        return PluginListResponse(
            plugins=plugins,
            total_count=total_count
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plugin type: {plugin_type}")
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail="Failed to list plugins")


@router.get("/metadata/{plugin_type}/{plugin_name}", response_model=PluginMetadataResponse)
async def get_plugin_metadata(
    plugin_type: str,
    plugin_name: str,
    user_id: str = Depends(get_current_user)
) -> PluginMetadataResponse:
    """
    Get metadata for a specific plugin.
    
    Args:
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin
        
    Returns:
        Plugin metadata
    """
    try:
        ptype = PluginType(plugin_type)
        metadata = PLUGIN_REGISTRY.get_plugin_metadata(plugin_name, ptype)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_name}")
        
        return PluginMetadataResponse(
            name=metadata.name,
            version=metadata.version,
            author=metadata.author,
            description=metadata.description,
            plugin_type=metadata.plugin_type.value,
            dependencies=metadata.dependencies,
            config_schema=metadata.config_schema,
            tags=metadata.tags
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plugin type: {plugin_type}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plugin metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to get plugin metadata")


@router.post("/load")
async def load_plugin(
    request: LoadPluginRequest,
    user_id: str = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Load and initialize a plugin instance.
    
    Args:
        request: Plugin loading request
        
    Returns:
        Success message
    """
    try:
        ptype = PluginType(request.plugin_type)
        
        # Check if plugin exists
        plugin_class = PLUGIN_REGISTRY.get_plugin_class(request.plugin_name, ptype)
        if not plugin_class:
            raise HTTPException(status_code=404, detail=f"Plugin not found: {request.plugin_name}")
        
        # Validate dependencies
        if not PLUGIN_LOADER.validate_dependencies(request.plugin_name, ptype):
            raise HTTPException(status_code=400, detail="Plugin dependencies not satisfied")
        
        # Create instance
        instance = PLUGIN_REGISTRY.create_instance(
            request.plugin_name,
            ptype,
            request.config
        )
        
        if not instance:
            raise HTTPException(status_code=500, detail="Failed to create plugin instance")
        
        logger.info(f"Loaded plugin: {request.plugin_name} for user {user_id}")
        
        return {"message": f"Plugin {request.plugin_name} loaded successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plugin type: {request.plugin_type}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading plugin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load plugin: {str(e)}")


@router.post("/use")
async def use_plugin(
    request: UsePluginRequest,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Use a loaded plugin to process data.
    
    Args:
        request: Plugin usage request
        
    Returns:
        Plugin execution results
    """
    try:
        ptype = PluginType(request.plugin_type)
        
        # Get or create plugin instance
        instance = PLUGIN_REGISTRY.create_instance(
            request.plugin_name,
            ptype,
            request.config
        )
        
        if not instance:
            raise HTTPException(status_code=404, detail=f"Plugin not found or failed to load: {request.plugin_name}")
        
        # Execute the requested method
        if not hasattr(instance, request.method):
            raise HTTPException(status_code=400, detail=f"Method not found: {request.method}")
        
        method = getattr(instance, request.method)
        
        # Convert data based on plugin type and method
        if ptype == PluginType.INFERENCE:
            if request.method == "extract_patterns":
                # Convert raw action data to Action objects
                actions = [Action(**a) for a in request.data.get("actions", [])]
                result = method(actions)
                # Result should be a list of pattern dictionaries
                return {"patterns": result}
                
            elif request.method == "infer_principles":
                # Pass pattern data directly (as dictionaries)
                patterns = request.data.get("patterns", [])
                result = method(patterns)
                # Convert Principle objects back to dicts
                return {"principles": [p.dict() for p in result]}
                
        elif ptype == PluginType.SCENARIO:
            if request.method == "generate_scenario":
                result = method(request.data.get("context", {}))
                return {"scenario": result}
                
            elif request.method == "generate_batch":
                count = request.data.get("count", 10)
                context = request.data.get("context", {})
                result = method(count, context)
                return {"scenarios": result}
                
        elif ptype == PluginType.ANALYSIS:
            if request.method == "analyze":
                result = method(request.data)
                return {"analysis": result}
                
            elif request.method == "generate_report":
                # Set export format if specified
                if "format" in request.data:
                    instance.config["current_format"] = request.data["format"]
                
                analysis_results = request.data.get("analysis_results", {})
                result = method(analysis_results)
                
                return {
                    "report": result,
                    "format": instance.config.get("current_format", "json")
                }
        
        # Generic method execution
        result = method(**request.data)
        
        # Handle different return types
        if hasattr(result, 'dict'):
            return {"result": result.dict()}
        elif isinstance(result, (list, dict, str, int, float, bool)):
            return {"result": result}
        else:
            return {"result": str(result)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plugin type: {request.plugin_type}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error using plugin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to use plugin: {str(e)}")


@router.post("/reload/{plugin_type}/{plugin_name}")
async def reload_plugin(
    plugin_type: str,
    plugin_name: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Reload a plugin (useful for development).
    
    Args:
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin
        
    Returns:
        Success message
    """
    try:
        ptype = PluginType(plugin_type)
        
        success = PLUGIN_LOADER.reload_plugin(plugin_name, ptype)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reload plugin")
        
        logger.info(f"Reloaded plugin: {plugin_name} for user {user_id}")
        
        return {"message": f"Plugin {plugin_name} reloaded successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plugin type: {plugin_type}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading plugin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload plugin: {str(e)}")


@router.get("/info")
async def get_plugin_system_info(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get information about the plugin system.
    
    Returns:
        Plugin system information
    """
    try:
        info = PLUGIN_LOADER.get_plugin_info()
        
        # Add registry statistics
        info["registry_stats"] = {
            "total_registered": sum(
                len(plugins) for plugins in PLUGIN_REGISTRY._plugins.values()
            ),
            "active_instances": len(PLUGIN_REGISTRY._instances)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting plugin system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get plugin system info")


@router.post("/discover")
async def discover_plugins(
    external_packages: Optional[List[str]] = Body(default=[]),
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Discover plugins from external packages.
    
    Args:
        external_packages: List of package names to search for plugins
        
    Returns:
        Discovery results
    """
    try:
        # Add external packages
        for package in external_packages:
            PLUGIN_LOADER.add_external_package(package)
        
        # Discover plugins
        discovered = PLUGIN_LOADER.discover_plugins()
        
        return {
            "discovered": discovered,
            "total_count": sum(len(p) for p in discovered.values())
        }
        
    except Exception as e:
        logger.error(f"Error discovering plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to discover plugins: {str(e)}")
