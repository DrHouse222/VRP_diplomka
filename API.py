import requests
import json
from typing import List, Dict, Any

from problem_types import VRP_PROBLEM_TYPE

# Short human-readable descriptions for each feature name
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Core spatial / demand features
    "dist_to_depot": "distance from (route) depot to candidate customer",
    "dist_from_current": "distance from current node to candidate customer",
    "savings": "Clarke-Wright style savings for inserting candidate",
    "demand": "demand of candidate customer",
    "remaining_capacity": "remaining vehicle capacity before adding candidate",
    "load_percentage": "current load divided by vehicle capacity",
    # Time-related features
    "arrival_time": "arrival time at candidate if selected next",
    "due_time": "latest allowed service start time at candidate",
    "wait_time": "waiting time before service can start at candidate",
    "slack_to_due": "time left until due time at arrival",
    "route_urgency": "slack_to_due divided by travel distance (higher = less urgent)",
    # Battery / green VRP features
    "current_battery": "current battery level of vehicle",
    "battery_percentage": "current_battery divided by battery capacity",
    "energy_to_customer": "energy required to reach candidate from current node",
    "dist_to_nearest_charger": "distance from candidate to nearest charging station",
    # Multi-depot features
    "depot_rank": "rank of this depot by distance to candidate (1 = closest depot)",
    "depot_distance_advantage": "how much closer this depot is vs second-closest depot to candidate",
}


def generate_vrp_heuristic(features_list, variant_description, api_key):
    """
    Use OpenRouter API to generate a VRP heuristic function.
    
    Args:
        strategy_description: What strategy to use (e.g., "greedy nearest neighbor")
        api_key: Your OpenRouter API key
    """
    
    prompt = f"""You are an expert in vehicle routing problems (VRP).

Create a scoring function for selecting the next customer in a VRP route construction algorithm.

**Variant** {variant_description}

**Context:**
- Lower score = better customer to visit next (minimization)
- Function will be called repeatedly to build routes sequentially

**Available features (as function parameters):** {features_list}

**Constants available:** 0.0, 1.0, -1.0

**Operations available:** add, sub, mul, protected_div, max, min

**Requirements:**
1. Function example: add(demand, protected_div(sub(dist_from_current, travel_time), protected_div(dist_to_depot, dist_from_current)))
2. Return a single float (the score)
3. Only use feature names that appear in the feature list above (do not invent new parameters).
4. Write the function on one line 

**Output ONLY the function, no explanation before or after.**
"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "arcee-ai/trinity-large-preview:free",
            "provider": {
                "order": ["Modal"]  # Force Modal provider
            },
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    )
    
    result = response.json()
    function_code = result['choices'][0]['message']['content']
    
    # Extract just the function (remove markdown if present)
    if "```python" in function_code:
        function_code = function_code.split("```python")[1].split("```")[0].strip()
    elif "```" in function_code:
        function_code = function_code.split("```")[1].split("```")[0].strip()
    
    return function_code


def get_features_for_variant(
    bool_capacity: bool, bool_TW: bool, bool_green: bool, bool_MD: bool
) -> List[str]:
    """Build the raw feature name list for a given variant."""
    features = [
        'dist_to_depot',
        'dist_from_current',
        'savings'
    ]
    if bool_capacity:
        features.append('demand')
        features.append('remaining_capacity')
        features.append('load_percentage')
    if bool_MD:
        features.append('depot_rank')
        features.append('depot_distance_advantage')
    if bool_TW:
        features.append('arrival_time')
        features.append('due_time')
        features.append('wait_time')
        features.append('slack_to_due')
        features.append('route_urgency')
    if bool_green:
        features.append('current_battery')
        features.append('battery_percentage')
        features.append('energy_to_customer')
        features.append('dist_to_nearest_charger')

    return features


def generate_all_vrp_heuristics(api_key: str) -> List[Dict[str, Any]]:
    """
    Call generate_vrp_heuristic 16 times, once for each (capacity, TW, green, MD)
    VRP variant used in our experiments.

    Returns a list of dicts with:
      - variant_name
      - bool_capacity, bool_TW, bool_green, bool_MD
      - features
      - heuristic_code
    """
    # Map (TW, green, MD) to variant name (as in DEAP_gen)
    variant_names: Dict[tuple, str] = {
        (False, False, False): "CVRP",
        (True,  False, False): "VRPTW",
        (False, True,  False): "GVRP",
        (True,  True,  False): "G-VRPTW",
        (False, False, True):  "MDCVRP",
        (True,  False, True):  "MDVRPTW",
        (False, True,  True):  "GVRP-MD",
        (True,  True,  True):  "G-VRPTW-MD",
    }

    results: List[Dict[str, Any]] = []

    for bool_capacity in (False, True):
        for bool_TW in (False, True):
            for bool_green in (False, True):
                for bool_MD in (False, True):
                    key = (bool_TW, bool_green, bool_MD)
                    if key not in variant_names:
                        continue

                    variant_name = variant_names[key]
                    features = get_features_for_variant(
                        bool_capacity=bool_capacity,
                        bool_TW=bool_TW,
                        bool_green=bool_green,
                        bool_MD=bool_MD,
                    )

                    # Build feature string with short descriptions for the prompt
                    entries: List[str] = []
                    for name in features:
                        desc = FEATURE_DESCRIPTIONS.get(name, "").strip()
                        if desc:
                            entries.append(f"{name}: {desc}")
                        else:
                            entries.append(name)
                    features_str = ", ".join(entries)

                    variant_description = (
                        f"{variant_name}: "
                        f"{'Multi Depot' if bool_MD else ''}, "
                        f"{'Green' if bool_green else ''} "
                        f"{'Vehicle Routing Problem with capacity constraints' if bool_capacity else ' Vehicle Routing Problem without capacity constraints'} "
                        f"{'with Time Windows' if bool_TW else ''}"
                    )


                    for i in range(20):
                        code = generate_vrp_heuristic(features_str, variant_description, api_key)

                        results.append(
                            {
                                "variant_name": variant_name,
                                "bool_capacity": bool_capacity,
                                "bool_TW": bool_TW,
                                "bool_green": bool_green,
                                "bool_MD": bool_MD,
                                "heuristic_code": code,
                            }
                        )

    return results

results = generate_all_vrp_heuristics(api_key="")
with open("generated_heuristics2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)