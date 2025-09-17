import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

# Charger le modèle
model_ref = bentoml.sklearn.get("house_price_model:latest")
model_runner = model_ref.to_runner()

# Créer le service
svc = bentoml.Service("house_price_predictor")

# Schéma d’entrée
class HouseInput(BaseModel):
    square_footage: float
    num_rooms: int

# API de prédiction (nouvelle syntaxe)
@svc.api_route("/predict", input=JSON(pydantic_model=HouseInput), output=JSON())
async def predict_house_price(data: HouseInput):
    input_data = [[data.square_footage, data.num_rooms]]
    prediction = await model_runner.async_run(input_data)
    return {"predicted_price": float(prediction[0])}
