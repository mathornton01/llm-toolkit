"""
Plugin registry for LLM Toolkit modules.

Usage:
    from llm_toolkit.core import Registry
    
    # Register a module
    @Registry.register("pruning", "wanda")
    class WandaPruning(PruningMethod):
        ...
    
    # Discover and instantiate
    cls = Registry.get("pruning", "wanda")
    pruner = cls(config)
    
    # List available
    Registry.list("pruning")  # ["magnitude", "wanda", "actmag", ...]
"""
from typing import Dict, Type, Optional, Any


class Registry:
    """Global registry for all toolkit modules."""
    
    _modules: Dict[str, Dict[str, Type]] = {}
    
    @classmethod
    def register(cls, category: str, name: str):
        """Decorator to register a module class.
        
        Args:
            category: Module category (pruning, finetuning, evaluation, etc.)
            name: Unique name within category
        """
        def decorator(klass):
            if category not in cls._modules:
                cls._modules[category] = {}
            if name in cls._modules[category]:
                raise ValueError(
                    f"Module '{name}' already registered in '{category}'. "
                    f"Existing: {cls._modules[category][name]}, New: {klass}"
                )
            cls._modules[category][name] = klass
            klass._registry_name = name
            klass._registry_category = category
            return klass
        return decorator
    
    @classmethod
    def get(cls, category: str, name: str) -> Type:
        """Get a registered module class."""
        if category not in cls._modules:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Available: {list(cls._modules.keys())}"
            )
        if name not in cls._modules[category]:
            raise KeyError(
                f"Unknown module '{name}' in '{category}'. "
                f"Available: {list(cls._modules[category].keys())}"
            )
        return cls._modules[category][name]
    
    @classmethod
    def list(cls, category: Optional[str] = None) -> Dict[str, list]:
        """List registered modules."""
        if category:
            return {category: sorted(cls._modules.get(category, {}).keys())}
        return {cat: sorted(mods.keys()) for cat, mods in cls._modules.items()}
    
    @classmethod
    def create(cls, category: str, name: str, **kwargs) -> Any:
        """Convenience: get class and instantiate with kwargs."""
        klass = cls.get(category, name)
        return klass(**kwargs)
    
    @classmethod
    def clear(cls):
        """Clear all registrations (for testing)."""
        cls._modules.clear()
