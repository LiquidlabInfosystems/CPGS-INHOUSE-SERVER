# cpgsapp/config.py

# Use a variable to store the ID once fetched
_cached_device_id = None

def get_device_id():
    """
    Safely retrieves the device ID from the database.
    Caches the result after the first successful fetch.
    Returns "default" if no account is found or if called too early.
    """
    global _cached_device_id

    # Return cached ID if available
    if _cached_device_id is not None:
        return _cached_device_id

    # Attempt to fetch from DB only when the function is called
    try:
        # Import models locally to avoid AppRegistryNotReady errors at module load time
        from django.apps import apps
        # Use apps.get_model to access the model safely after apps are ready
        Account = apps.get_model('cpgsapp', 'Account')

        device_account = Account.objects.first()

        if device_account and device_account.device_id:
            _cached_device_id = str(device_account.device_id)
            print(f"Device ID loaded from DB: {_cached_device_id}")
        else:
            _cached_device_id = "default"
            print("No Account found or device_id is empty. Using default Device ID: default")

    except LookupError:
        # This exception is raised by apps.get_model if apps aren't ready
        print("Warning: Attempted to get device ID before app registry was fully ready. Using default.")
        _cached_device_id = "default" # Fallback in case called too early
    except Exception as e:
        # Catch other potential database errors
        print(f"Error fetching device ID from database: {e}. Using default.")
        _cached_device_id = "default" # Fallback

    return _cached_device_id

# You can add other global configuration variables or functions here later
# Example:
# SOME_OTHER_SETTING = "some_value"
