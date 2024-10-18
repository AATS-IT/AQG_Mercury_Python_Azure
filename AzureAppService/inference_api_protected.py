import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
import os

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "9205a116f4ad4ccc76412751cc83c71bd90116bc045584be25597b78ff3ccf5b"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# This is a mock user database. In a real application, you'd use a proper database.
def get_password_hash(password):
    return pwd_context.hash(password)

fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("admin"),
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str

class UserInDB(User):
    hashed_password: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Load models and optimal threshold data once on startup
models = joblib.load('models/all_lite_classweighting_model.pkl')
prevalence_df = pd.read_csv('models/class_prevalence.csv')

# Lite models dictionary adjusted based on available variables from the CSV
lite_models = {
    'opdead': {
        'predictors': ["albmn_pr", "aodis_pr", "fev1pct", "hgb_pr", "ht", "lvef_pr", "meldscor", "nyha_pr", 
                       "pasys_pr", "pr_count", "srg_elct", "srg_emrg", "surgprim", "thao_pr", "tvrgsev", "wbc_pr"],
        'response': 'opdead'
    },
    'losopg14': {
        'predictors': ["albmn_pr", "aodis_pr", "fev1pct", "hgb_pr", "ht", "los_pre", "lvef_pr", "med_inot", 
                       "meldscor", "nyha_pr", "pr_count", "srg_elct", "srg_emrg", "surg_num", "tvrgsev", "wbc_pr"],
        'response': 'losopg14'
    },
    'ev_print': {
        'predictors': ["albmn_pr", "aodis_pr", "fev1pct", "hgb_pr", "ht", "lvef_pr", "meldscor", "nyha_pr", 
                       "pasys_pr", "pr_count", "srg_elct", "srg_emrg", "surgprim", "thao_pr", "tvrgsev", "wbc_pr", 
                       "wtht"],
        'response': 'ev_print'
    }, 
    'ev_dsinf': {
        'predictors': ["albmn_pr", "bmi", "chfacute", "creat_pr", "hgb_pr", "ht", "los_pre", "lvedsi", "lvef_pr", 
                       "meldscor", "pasys_pr", "pr_count", "wbc_pr", "wtht"],
        'response': 'ev_dsinf'
    },
    'ev_strok': {
        'predictors': ["albmn_pr", "creat_pr", "cshocprc", "hgb_pr", "ht", "hx_cvd", "hx_thao", "lvef_pr", 
                       "meldscor", "pltlt_pr", "pr_count", "srg_elct", "srg_emrg", "thao_pr", "wbc_pr"],
        'response': 'ev_strok'
    },
    'ev_reop': {
        'predictors': ["albmn_pr", "bmi", "cad_sys", "creat_pr", "fev1pct", "hgb_pr", "los_pre", "lvedsi", "med_inot", 
                       "meldscor", "nyha_pr", "pasys_pr", "pltlt_pr", "pr_count",  "srg_elct", 
                       "srg_emrg", "surg_num", "tvrgsev", "wtht"],
        'response': 'ev_reop'
    },
    'ev_renal': {
        'predictors': ["albmn_pr", "aodis_pr", "chfacute", "creat_pr", "fev1pct", "hgb_pr", "ht", "hx_pad", 
                       "iabp_pr", "meldscor", "pasys_pr", "pr_count", "srg_elct", "srg_emrg", 
                       "tvrgsev", "wbc_pr"],
        'response': 'ev_renal'
    },
    'ev_mm': {
        'predictors': ["albmn_pr", "aodis_pr", "creat_pr", "cshocprc", "fev1pct", "hgb_pr", "ht", "hx_pad", 
                       "hx_thao", "iabp_pr", "lvef_pr", "meldscor", "nyha_pr", "pasys_pr", "pr_count", "srg_elct", 
                       "srg_emrg", "surgprim", "thao_pr", "tvrgsev", "wbc_pr", "wtht"],
        'response': 'ev_mm'
    }
}

# Directory to save temporary files
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Function to process the uploaded CSV and generate predictions
def run_inference_on_csv(file_path: str) -> str:
    # Load the input data
    new_data = pd.read_csv(file_path)

    # Create a copy of the new_data to store predictions
    predictions_df = new_data.copy()

    # Run inference and add probability and class prediction columns for each model
    for model_name, model_info in lite_models.items():
        predictors = model_info['predictors']

        # Ensure all required predictors are present in the new data
        available_predictors = [pred for pred in predictors if pred in new_data.columns]
        
        # If not all required predictors are available, skip this model
        if len(available_predictors) < len(predictors):
            print(f"Skipping {model_name} due to missing predictors.")
            continue

        # Extract relevant features from new data
        X_new = new_data[available_predictors]

        # Load the corresponding pre-trained model
        model = models[model_name]

        # Get predicted probabilities and class based on the optimal threshold
        y_pred_proba = model.predict_proba(X_new)[:, 1]

        # Get the optimal threshold (prevalence) for this model from the CSV
        optimal_threshold = prevalence_df[prevalence_df['Model'] == model_name]['Optimal_Threshold'].values[0]
        
        # Get class predictions using the optimal threshold
        y_pred_class = (y_pred_proba >= optimal_threshold).astype(int)

        # Add probability and class predictions to the dataframe
        predictions_df[f'{model_name}_probability'] = y_pred_proba
        predictions_df[f'{model_name}_prediction'] = y_pred_class

    # Save the output with predictions to a new CSV file
    output_file_path = os.path.join(TEMP_DIR, 'model_predictions_with_probabilities.csv')
    predictions_df.to_csv(output_file_path, index=False)
    
    return output_file_path

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run inference on the uploaded CSV
    output_file = run_inference_on_csv(file_path)

    # Return the processed CSV with predictions for download
    return FileResponse(output_file, filename="model_predictions_with_probabilities.csv")

@app.post("/upload_csv_protected")
async def upload_csv_protected(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run inference on the uploaded CSV
        output_file = run_inference_on_csv(file_path)

        # Return the processed CSV with predictions for download
        return FileResponse(output_file, filename="model_predictions_with_probabilities.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up: remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

# uvicorn inference_api_protected:app --reload
