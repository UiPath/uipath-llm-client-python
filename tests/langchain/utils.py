from langchain_core.tools import tool


@tool
def search_accommodation(location: str, check_in: str, check_out: str) -> str:
    """Search for available accommodation in a location for given dates."""
    return f"Found 5 hotels in {location} from {check_in} to {check_out}: Hotel A ($100/night), Hotel B ($150/night), Hotel C ($200/night)"


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date."""
    return f"Found 3 flights from {origin} to {destination} on {date}: Flight 1 (9:00 AM, $300), Flight 2 (2:00 PM, $250), Flight 3 (7:00 PM, $350)"


@tool
def search_attractions(location: str) -> str:
    """Search for tourist attractions and activities in a location."""
    return f"Top attractions in {location}: Museum of Art, Central Park, Historic District, Local Food Tour, Beach Activities"
