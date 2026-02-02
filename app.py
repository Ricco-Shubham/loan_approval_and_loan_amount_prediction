import joblib
import os
import pandas as pd
import streamlit as st


class LoanApprovalApp:
    def __init__(self,classifier="stage_1_rf_classifier_pipeline.pkl", regressor = "stage_2_rf_regression_pipeline.pkl"):
        # classifier : stage 1 model (0/1 prediction)
        # regressor  : stage 2 regression model
        model_path = os.path.join(os.getcwd(),'models')

        self.clf = joblib.load(os.path.join(model_path,classifier))
        self.reg = joblib.load(os.path.join(model_path,regressor))

        self.col = [
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]

    def get_user_input(self):

        data = {
            "no_of_dependents": st.sidebar.number_input("Dependents",0,10,0),
            "education" :st.sidebar.selectbox(
                "Education", ["Graduate","Not Graduate"]
            ),
            "self_employed": st.sidebar.selectbox(
                "Self Employed",["Yes","No"]
            ),
            "income_annum" : st.sidebar.number_input("Annual Income",value=0.0),
            "loan_term" : st.sidebar.number_input("Loan Term",value=0),
            "cibil_score" : st.sidebar.number_input("cibil score",value=0),
            "residential_assets_value" : st.sidebar.number_input("residential assets value",value=0),
            "commercial_assets_value" : st.sidebar.number_input("commercial assets value",value=0),
            "luxury_assets_value" : st.sidebar.number_input("luxury assets value",value=0),
            "bank_asset_value" : st.sidebar.number_input("bank asset value",value=0),
            "loan_amount" : st.sidebar.number_input("loan amount",value=0),
        }

        applicant_df = pd.DataFrame([data])
        return applicant_df


    
    def two_stage_predictor(self,applicant_df):
        out = {}

        # stage 1: Loan Aprroval
        approve = self.clf.predict(applicant_df)[0]
        out['loan_status'] = int(approve)

        if approve == 1:
            print("Your Loan application is aprroved by the ML system")
            applicant_df_reg = applicant_df.drop(columns=['loan_amount']).copy()
            applicant_df_reg['loan_status'] = 'approve'

            # stage 2: prediction loan amount
            loan_amount_pred = self.reg.predict(applicant_df_reg)[0]
            out['predicted_loan_amount'] = float(loan_amount_pred)
        else:
            out['predicted_loan_amount'] = None
        return out

#-----------------STREAMLIT UI----------------#

if __name__ == "__main__":
    # used streamlit 
    st.set_page_config(page_title="Loan Predictor",layout="centered")
    st.title("🏦 Loan Approval & Amount Predictor")
    st.write("Fill applicant details below")

    #sidebar Inputs
    st.sidebar.header("Applicant Details")
    app = LoanApprovalApp()
    applicant_df = app.get_user_input()
    
    #predict Button

    if st.button("🔍 Predict Loan Status"):
        result = app.two_stage_predictor(applicant_df)
        st.subheader("Result")

        if result['loan_status']==1:
            st.success("✅ Loan Approved")
            st.metric(
                "💰 Predicted Loan Amount",
                f"₹ {result['predicted_loan_amount']:,.2f}",
            )

        else:

            st.error("Loan Rejected")

