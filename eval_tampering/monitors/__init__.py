"""Lazy JSON entry point for offline and text monitors."""

from ..messages import InputError, failure, validate_request


def handle(request):
    try:
        operation = request.get('operation') if type(request) is dict else None
        if type(operation) is str and operation.startswith(('text.', 'reasoning.')):
            if operation.startswith('text.'):
                from .text_monitor import TextMonitor as Component, OPERATIONS
            else:
                from .reasoning_annotator import ReasoningAnnotator as Component, OPERATIONS
            validate_request(request, OPERATIONS)
            monitor = Component(request['config'])
            provider = monitor.config['provider']
            if operation in ('text.score', 'reasoning.annotate') and provider is not None:
                from openai import OpenAI, OpenAIError
                try:
                    with OpenAI(max_retries=0, timeout=provider['timeout_seconds'], base_url='https://api.openai.com/v1') as client:
                        return Component(request['config'], client).handle(request)
                except OpenAIError:
                    return failure(request, InputError('provider_unavailable', 'OpenAI client could not be initialized or completed; check its configuration and saved records'))
            return monitor.handle(request)
        else:
            from .activation_monitor import ActivationMonitor, OPERATIONS
            validate_request(request, OPERATIONS)
            return ActivationMonitor(request['config']).handle(request)
    except (InputError, OSError, ImportError) as exc:
        return failure(request, exc if isinstance(exc, InputError) else InputError('dependency_error' if isinstance(exc, ImportError) else 'file_error', str(exc)))
