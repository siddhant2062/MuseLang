"""
Inject a minimal triton stub when triton is not installed (e.g. macOS).
Enables muselang and dependencies (transformers, torchao) to import without errors.
Uses lazy submodules so any triton.* or triton.*.* import works (e.g. triton.compiler.compiler).
"""
import sys

if "triton" not in sys.modules:
    try:
        import triton  # noqa: F401
    except ImportError:
        import types
        from importlib.machinery import ModuleSpec

        _triton_dtype = type("triton_dtype", (), {})
        _triton_lang_spec = ModuleSpec("triton.language", None, origin="(muselang triton stub)")
        _STUB_ORIGIN = "<triton stub: no wheel on this platform>"

        class _TritonLangStub(types.ModuleType):
            dtype = _triton_dtype
            __file__ = _STUB_ORIGIN

            def __getattr__(self, name):
                if name in ("__file__", "__package__"):
                    return _STUB_ORIGIN if name == "__file__" else "triton.language"
                return _triton_dtype

        _triton_lang = _TritonLangStub("triton.language")
        _triton_lang.__spec__ = _triton_lang_spec
        _triton_lang.__file__ = _STUB_ORIGIN

        class _TritonConfig:
            def __init__(self, *args, **kwargs):
                pass

        def _noop_decorator(fn=None, **kwargs):
            def dec(f):
                return f
            return dec if fn is None else dec(fn)

        def _noop(*args, **kwargs):
            return 1 if not args else args[0]

        def _make_lazy_submodule(full_name: str) -> types.ModuleType:
            """Create a stub submodule; any attribute access returns another stub submodule."""
            mod = types.ModuleType(full_name)
            mod.__file__ = _STUB_ORIGIN
            mod.__package__ = full_name
            mod.__path__ = []

            def __getattr__(name: str):
                if name in ("__file__", "__path__", "__package__"):
                    return _STUB_ORIGIN if name != "__package__" else full_name
                child_name = f"{full_name}.{name}"
                if child_name not in sys.modules:
                    child = _make_lazy_submodule(child_name)
                    sys.modules[child_name] = child
                    setattr(mod, name, child)
                return getattr(mod, name)

            mod.__getattr__ = __getattr__
            return mod

        class _TritonStub(types.ModuleType):
            def __getattr__(self, name):
                if name in ("__file__", "__path__"):
                    return _STUB_ORIGIN
                if name == "__package__":
                    return "triton"
                if name == "__doc__":
                    return "Triton stub (no GPU kernel support on this platform)."
                # Lazy submodule: any triton.foo.bar... import works
                full_name = f"triton.{name}"
                if full_name not in sys.modules:
                    sub = _make_lazy_submodule(full_name)
                    sys.modules[full_name] = sub
                    setattr(self, name, sub)
                return getattr(self, name)

        _triton_stub = _TritonStub("triton")
        _triton_stub.__path__ = []
        _triton_stub.__file__ = _STUB_ORIGIN
        _triton_stub.__package__ = "triton"
        _triton_stub.__version__ = "0.0.0.stub"  # torch.utils._triton.get_triton_version() may read this
        _triton_stub.__spec__ = ModuleSpec("triton", None, origin=_STUB_ORIGIN)
        _triton_stub.language = _triton_lang
        _triton_stub.Config = _TritonConfig
        _triton_stub.jit = _noop_decorator
        _triton_stub.autotune = _noop_decorator
        _triton_stub.cdiv = _noop
        _triton_stub.next_power_of_2 = _noop
        _triton_stub.reinterpret = _noop

        # Pre-register submodules so "import triton.backends.compiler" / "import triton.compiler.compiler"
        # work: the import system looks up sys.modules by full name and does not always call __getattr__.
        for _sub in ("backends", "compiler"):
            _submod = _make_lazy_submodule(f"triton.{_sub}")
            sys.modules[f"triton.{_sub}"] = _submod
            setattr(_triton_stub, _sub, _submod)
        # torch.utils._triton.cpu_extra_check does "cpu" in triton.backends.backends; must be iterable.
        sys.modules["triton.backends"].backends = []
        for _parent, _child in (("triton.backends", "compiler"), ("triton.compiler", "compiler")):
            _submod = _make_lazy_submodule(f"{_parent}.{_child}")
            sys.modules[f"{_parent}.{_child}"] = _submod
            setattr(sys.modules[_parent], _child, _submod)

        sys.modules["triton"] = _triton_stub
        sys.modules["triton.language"] = _triton_lang
