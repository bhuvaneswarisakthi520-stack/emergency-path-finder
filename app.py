from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import folium
import networkx as nx
import numpy as np
import osmnx as ox
ox.settings.overpass_url = "https://overpass.kumi.systems/api"
ox.settings.overpass_rate_limit = False
ox.settings.requests_timeout = 180
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from ai_modules import RiskIntelligenceEngine


st.set_page_config(
    page_title="AI Emergency Path Finder",
    page_icon="🚑",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "US_Accidents_Cleaned_900.csv"


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * radius * atan2(sqrt(value), sqrt(1 - value))


@st.cache_data
def load_hotspots():
    data = pd.read_csv(DATA_FILE)

    data["Lat_Grid"] = data["Start_Lat"].round(2)
    data["Lng_Grid"] = data["Start_Lng"].round(2)

    hotspots = data.groupby(
        ["Lat_Grid", "Lng_Grid"],
        as_index=False
    ).agg(
        Accident_Frequency=("ID", "count"),
        Average_Severity=("Severity", "mean")
    )

    hotspots["Risk_Score"] = (
        hotspots["Accident_Frequency"]
        * hotspots["Average_Severity"]
    )

    return hotspots


@st.cache_resource(show_spinner="Training the prototype spatial-temporal GAT model...")
def load_risk_engine():
    """Build the unified emergency spatial dataset and train prototype AI models."""
    return RiskIntelligenceEngine(pd.read_csv(DATA_FILE))


def add_risk_weights(road_graph, risk_engine):
    for node, attributes in road_graph.nodes(data=True):
        latitude = attributes["y"]
        longitude = attributes["x"]

        # Spatial-temporal GAT prediction projected from accident grids to roads.
        road_graph.nodes[node]["risk_score"] = risk_engine.road_risk(latitude, longitude)

    for start, end, key, edge in road_graph.edges(keys=True, data=True):
        average_risk = (
            road_graph.nodes[start]["risk_score"]
            + road_graph.nodes[end]["risk_score"]
        ) / 2

        edge["risk_weight"] = edge["length"] * (
            1 + average_risk * 2
        )

    return road_graph


def route_metrics(graph, route, weight_name):
    distance_meters = 0
    risk_total = 0

    for start, end in zip(route[:-1], route[1:]):
        edge_options = graph.get_edge_data(start, end)

        edge = min(
            edge_options.values(),
            key=lambda item: item.get(weight_name, float("inf"))
        )

        distance_meters += edge["length"]

        risk_total += (
            graph.nodes[start]["risk_score"]
            + graph.nodes[end]["risk_score"]
        ) / 2

    # Emergency vehicle prototype ETA: average speed 40 km/h
    eta_minutes = (distance_meters / 1000) / 40 * 60

    return round(distance_meters / 1000, 2), round(eta_minutes, 1), round(risk_total, 2)


def route_coordinates(graph, route):
    return [
        [graph.nodes[node]["y"], graph.nodes[node]["x"]]
        for node in route
    ]


def adaptive_risk_aware_astar(graph, start_node, end_node):
    """Stage 4 adaptive A*: distance heuristic with dynamic learned-risk cost."""
    end_latitude = graph.nodes[end_node]["y"]
    end_longitude = graph.nodes[end_node]["x"]

    def heuristic(node, _target):
        return haversine_km(
            graph.nodes[node]["y"], graph.nodes[node]["x"],
            end_latitude, end_longitude
        ) * 1000

    def adaptive_cost(_start, _end, edge_data):
        # NetworkX supplies all parallel edges for an OSMnx MultiDiGraph.
        if "risk_weight" in edge_data:
            return edge_data["risk_weight"]
        return min(item["risk_weight"] for item in edge_data.values())

    return nx.astar_path(
        graph, start_node, end_node, heuristic=heuristic, weight=adaptive_cost
    )


st.title("🚑 AI-Based Emergency Path Finder")
st.caption(
    "Spatial-temporal graph intelligence for real-road emergency routing"
)

st.info(
    "Review 2 prototype: enter two locations inside the Greater Columbus, Ohio area."
)

left, right = st.columns(2)

with left:
    origin = st.text_input(
        "Emergency vehicle starting location",
        "The Ohio State University, Columbus, Ohio"
    )

with right:
    destination = st.text_input(
        "Emergency destination",
        "John Glenn Columbus International Airport, Ohio"
    )

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if st.button("Find Emergency Route", type="primary"):
    st.session_state.show_result = True

if st.session_state.show_result:
    try:
        with st.spinner("Finding locations and downloading the road network..."):
            start_lat, start_lng = ox.geocoder.geocode(origin)
            end_lat, end_lng = ox.geocoder.geocode(destination)

            direct_distance = haversine_km(
                start_lat, start_lng,
                end_lat, end_lng
            )

            if direct_distance > 35:
                st.error(
                    "For this Review 2 prototype, choose two locations within "
                    "35 km of each other in Greater Columbus, Ohio."
                )
                st.stop()

            midpoint = (
                (start_lat + end_lat) / 2,
                (start_lng + end_lng) / 2
            )

            radius_meters = max(
                5000,
                int((direct_distance * 1000) / 2 + 5000)
            )

            road_graph = ox.graph_from_point(
                midpoint,
                dist=radius_meters,
                network_type="drive"
            )

            hotspots = load_hotspots()
            risk_engine = load_risk_engine()
            road_graph = add_risk_weights(road_graph, risk_engine)

            start_node = ox.distance.nearest_nodes(
                road_graph,
                X=start_lng,
                Y=start_lat
            )

            end_node = ox.distance.nearest_nodes(
                road_graph,
                X=end_lng,
                Y=end_lat
            )

            shortest_route = ox.routing.shortest_path(
                road_graph,
                start_node,
                end_node,
                weight="length"
            )

            safest_route = adaptive_risk_aware_astar(
                road_graph, start_node, end_node
            )

        shortest_distance, shortest_eta, shortest_risk = route_metrics(
            road_graph, shortest_route, "length"
        )

        safest_distance, safest_eta, safest_risk = route_metrics(
            road_graph, safest_route, "risk_weight"
        )

        result_table = pd.DataFrame({
            "Route": ["Shortest Route", "Risk-Aware Route"],
            "Distance (km)": [shortest_distance, safest_distance],
            "Estimated ETA (minutes)": [shortest_eta, safest_eta],
            "Accident Risk Score": [shortest_risk, safest_risk]
        })
        # Stage 5: Monte Carlo Dropout uncertainty and route reliability.
        route_node_risks = [
            road_graph.nodes[node]["risk_score"]
            for node in safest_route
        ]   

        all_node_risks = [
            attributes["risk_score"]
            for _, attributes in road_graph.nodes(data=True)
        ]

        low_cutoff, high_cutoff = np.quantile(all_node_risks, [0.33, 0.66])
        average_route_risk = np.mean(route_node_risks)

        if average_route_risk <= low_cutoff:
            risk_level = "Low"
        elif average_route_risk <= high_cutoff:
            risk_level = "Medium"
        else:
            risk_level = "High"

        mc_uncertainty = risk_engine.route_uncertainty(road_graph, safest_route)
        confidence_score = round(max(50, min(99, (1 - mc_uncertainty) * 100)), 1)

        # Stage 4: DQN-based candidate route selection.
        dqn_choice, q_values = risk_engine.route_decision(
            shortest_distance, shortest_risk, safest_distance, safest_risk
        )
        if safest_risk < shortest_risk:
            recommended_route = "Risk-Aware Route"
        else:
            recommended_route = dqn_choice

        st.subheader("Emergency Route Recommendation")

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric("Recommended Route", recommended_route)
        metric_2.metric("Route Risk Level", risk_level)
        metric_3.metric(
            "MC Dropout Confidence",
            f"{confidence_score}%"
        )

        if safest_risk < shortest_risk:
            st.success(
                "Recommendation: Use the Risk-Aware Route because it has a lower "
                "historical accident-risk score than the shortest route."
            )

        st.subheader("Accident Hotspot Summary")

        top_hotspots = hotspots.nlargest(
            5,
            "Risk_Score"
        )[
            [
                "Lat_Grid",
                "Lng_Grid",
                "Accident_Frequency",
                "Average_Severity",
                "Risk_Score"
            ]
        ]

        st.dataframe(top_hotspots, use_container_width=True)

        st.subheader("AI Prototype Intelligence")
        ai_left, ai_middle, ai_right = st.columns(3)
        ai_left.metric("GAT Risk Locations", len(risk_engine.locations))
        ai_middle.metric("MC Uncertainty", f"{mc_uncertainty:.3f}")
        ai_right.metric("DQN Route Policy", recommended_route)

        top_features, feature_scores = risk_engine.explanation()
        xai_table = pd.DataFrame({
            "Graph XAI risk factor": top_features,
            "Relative contribution (%)": feature_scores,
        })
        st.caption(
            "Spatial-temporal GAT learns location risk; Monte Carlo Dropout estimates "
            "uncertainty; the DQN scores the two candidate routes."
        )
        st.dataframe(xai_table, use_container_width=True, hide_index=True)

        st.subheader("Route Comparison")
        st.dataframe(result_table, use_container_width=True)

        shortest_points = route_coordinates(road_graph, shortest_route)
        safest_points = route_coordinates(road_graph, safest_route)

        route_map = folium.Map(
            location=shortest_points[0],
            zoom_start=12,
            tiles="OpenStreetMap"
        )

        folium.Marker(
            shortest_points[0],
            tooltip="Start",
            icon=folium.Icon(color="blue", icon="play")
        ).add_to(route_map)

        folium.Marker(
            shortest_points[-1],
            tooltip="Destination",
            icon=folium.Icon(color="black", icon="flag")
        ).add_to(route_map)

        folium.PolyLine(
            safest_points,
            color="green",
            weight=6,
            tooltip="Risk-Aware Route"
        ).add_to(route_map)

        folium.PolyLine(
            shortest_points,
            color="red",
            weight=4,
            dash_array="10, 10",
            tooltip="Shortest Route"
        ).add_to(route_map)

        st.subheader("Shortest Route vs Risk-Aware Route")
        st_folium(route_map, width=1100, height=600)

        risk_reduction = shortest_risk - safest_risk

        st.subheader("Route Explanation")
        st.write(
            f"The risk-aware route changes the path using historical accident "
            f"hotspots and learned graph-risk estimates. It reduces the route-risk "
            f"score by **{risk_reduction:.2f}**. The strongest explanation factors "
            f"are **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**."
        )

        with st.expander("Prototype pipeline implementation status"):
            st.markdown("""
            - **Stage 1:** US Accidents preprocessing, feature selection, and coordinate grids.
            - **Stage 2:** Dynamic OpenStreetMap urban road graph with risk-based edge weights.
            - **Stage 3:** Trained lightweight spatial-temporal Graph Transformer / Graph Attention Network (GAT) and hotspot analysis.
            - **Stage 4:** Adaptive risk-aware cost, Adaptive A* routing, plus a prototype DQN candidate-route policy.
            - **Stage 5:** Monte Carlo Dropout uncertainty and feature-contribution Graph XAI.
            - **Stage 6:** Route recommendation, distance, ETA, risk level, hotspots, confidence, and explanation.

            This is a small-dataset academic prototype, not a validated emergency-dispatch system.
            """)

    except Exception as error:
        st.error(f"Unable to calculate the route: {error}")
        st.info(
            "Check that both locations are valid Greater Columbus addresses "
            "and that your internet connection is active."
        )
