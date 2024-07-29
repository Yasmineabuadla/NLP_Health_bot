import pandas as pd
import networkx as nx

def create_knowledge_graph(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Create an empty graph
    G = nx.Graph()

    # Iterate through the DataFrame and add nodes and edges
    for index, row in df.iterrows():
        patient_id = f"Patient_{index}"
        disease = row['Disease']
        age = row['Age']
        gender = row['Gender']
        blood_pressure = row['Blood Pressure']
        cholesterol_level = row['Cholesterol Level']
        outcome = row['Outcome Variable']

        # Symptoms
        symptoms = {
            'Fever': row['Fever'],
            'Cough': row['Cough'],
            'Fatigue': row['Fatigue'],
            'Difficulty Breathing': row['Difficulty Breathing']
        }

        # Add patient node
        G.add_node(patient_id, type='Patient', age=age, gender=gender, blood_pressure=blood_pressure, cholesterol_level=cholesterol_level, outcome=outcome)

        # Add disease node and edge
        G.add_node(disease, type='Disease')
        G.add_edge(patient_id, disease, relation='has_disease')

        # Add symptom nodes and edges
        for symptom, present in symptoms.items():
            if present == 'Yes':
                G.add_node(symptom, type='Symptom')
                G.add_edge(patient_id, symptom, relation='has_symptom')

    return G

def query_knowledge_graph(graph, entity_type, entity_value):
    print(f"Querying graph for {entity_type} = {entity_value}")
    result = []

    if entity_type == 'Patient':
        # Find all symptoms and diseases for a given patient
        neighbors = list(graph.neighbors(entity_value))
        for neighbor in neighbors:
            if graph.nodes[neighbor]['type'] == 'Symptom':
                result.append(f"Symptom: {neighbor}")
            elif graph.nodes[neighbor]['type'] == 'Disease':
                result.append(f"Disease: {neighbor}")

    elif entity_type == 'Disease':
        # Find all symptoms for a specific disease
        symptoms = set()
        for node in graph.nodes:
            if graph.nodes[node]['type'] == 'Patient':
                neighbors = list(graph.neighbors(node))
                if entity_value in neighbors:
                    for neighbor in neighbors:
                        if graph.nodes[neighbor]['type'] == 'Symptom':
                            symptoms.add(neighbor)
        result.extend([f"Symptom: {symptom}" for symptom in symptoms])

    elif entity_type == 'Symptom':
        # Find all patients with a specific symptom
        for node in graph.nodes:
            if graph.nodes[node]['type'] == 'Patient':
                neighbors = list(graph.neighbors(node))
                if entity_value in neighbors:
                    result.append(f"Patient: {node}")

    print(f"Query result: {result}")
    return result



### Meta data about the graph
#
# # Print the graph information
# num_nodes = G.number_of_nodes()
# num_edges = G.number_of_edges()
# graph_info = f'Graph with {num_nodes} nodes and {num_edges} edges'
# print(graph_info)


def convert_query_to_knowledge_graph_format(query):
    query_lower = query
    if "patient" in query_lower:
        if "which patients have disease" in query_lower:
            disease_name = query_lower.split("which patients have disease")[1].strip()
            return {'entity_type': 'Disease', 'entity_value': disease_name}
        elif "which patients have the symptom" in query_lower:
            symptom_name = query_lower.split("which patients have the symptom")[1].strip()
            return {'entity_type': 'Symptom', 'entity_value': symptom_name}
        else:
            patient_id = query_lower.split("patient")[1].strip().replace(" ", "_")
            return {'entity_type': 'Patient', 'entity_value': f'Patient_{patient_id}'}
    elif "disease" in query_lower:
        if "what are the symptoms of disease" in query_lower:
            disease_name = query_lower.split("what are the symptoms of disease")[1].strip()
            return {'entity_type': 'Disease', 'entity_value': disease_name}
        else:
            disease_name = query_lower.split("disease")[1].strip()
            return {'entity_type': 'Disease', 'entity_value': disease_name}
    elif "symptom" in query_lower:
        symptom_name = query_lower.split("symptom")[1].strip()
        return {'entity_type': 'Symptom', 'entity_value': symptom_name}
    else:
        return None