# Fair Lending and Non-Discrimination Standard

# ASHVA CAPITAL INTERNAL CREDIT POLICY MANUAL
## SECTION 4.0: FAIR LENDING AND NON-DISCRIMINATION STANDARD
### DOCUMENT ID: ACP-FL-402-REV.4
### EFFECTIVE DATE: OCTOBER 1, 2023

---

### 4.1 PURPOSE AND SCOPE
This section establishes the mandatory fair lending requirements for all consumer and commercial credit underwriting decisions executed within Ashva Capital ("the Institution"). Compliance with this policy is absolute. Underwriters, automated decision engines, and risk management personnel are strictly bound by the rules contained herein. Failure to comply constitutes a severe violation of employment terms and regulatory standards.

---

### 4.2 PROHIBITED PROTECTED ATTRIBUTES
Under no circumstances shall any credit decision, risk scoring model, pricing determination, or credit limit assignment consider, ingest, or weigh the following protected attributes of an applicant, co-applicant, guarantor, or director:

1. **Religion or Creed:** Affiliation with, practice of, or perceived membership in any religious group or denomination.
2. **Caste, Tribe, or Social Stratum:** Any categorization based on traditional social stratification, lineage, clan, or ancestral tribal identity.
3. **Gender, Sex, or Sexual Orientation:** Biological sex, gender identity, gender expression, or sexual orientation.
4. **Marital Status:** Whether an applicant is single, married, divorced, separated, widowed, or in a domestic partnership.
5. **Age:** Chronological age of the applicant, except for the verification of legal age of majority (minimum 18 years) required to enter into a binding contract. Age-based credit scoring adjustments are strictly prohibited.
6. **Disability or Medical Status:** Any physical, sensory, intellectual, or mental impairment, or medical history.
7. **Region of Origin, Place of Birth, or Nationality:** Geographic origin within or outside the operating jurisdiction, including ancestral home regions or national origin.

---

### 4.3 PROHIBITED PROXY VARIABLES
To prevent indirect discrimination, the ingestion or evaluation of variables that serve as functional proxies for the protected attributes listed in Section 4.2 is prohibited. The following variables are specifically banned from all credit assessment algorithms, decision trees, and manual underwriting files:

1. **Postal Code Alone:** Standalone postal codes, ZIP codes, or sector codes shall not be used as risk predictors. (Geographic risk must be evaluated at the municipal or state level in conjunction with verified macroeconomic data, and only where statistical correlation with protected classes has been formally disproved).
2. **Applicant First Name:** Natural language processing of applicant first names, honorifics, or paternal lineages for the purpose of inferring religion, caste, ethnicity, or gender.
3. **Institution of Education:** The specific name, tier, or type of high school, college, or university attended by the applicant.
4. **Language of Application:** The primary language, script, or dialect selected by the applicant to complete the digital application form.

---

### 4.4 DISPARATE-IMPACT TESTING REQUIREMENTS
1. **Mandatory Cadence:** The Risk Analytics and Fair Lending Department, under the direct oversight of Chief Risk Officer Marcus Vance, shall execute disparate-impact testing on all active credit scoring models, underwriting rulesets, and automated decision pipelines on a strict **quarterly basis** (due on the 15th of January, April, July, and October).
2. **Statistical Threshold:** Disparate impact shall be measured using the Four-Fifths (80%) Rule. If the selection rate for any protected class defined in Section 4.2 is less than 80% of the selection rate of the highest-retained control group, the model or ruleset fails the test.
3. **Remediation Protocol:** Upon failure of a quarterly disparate-impact test, the affected automated decision pipeline must be suspended within **24 hours** and reverted to the pre-approved deterministic fallback ruleset until the underlying feature weights are recalibrated and re-tested.

---

### 4.5 INVALIDATION OF DECISION RATIONALE
1. **Zero-Tolerance Rule:** Any internal underwriting memo, automated decision log, or customer communication rationale that explicitly references, alludes to, or incorporates any protected attribute (Section 4.2) or prohibited proxy variable (Section 4.3) is **void ab initio**.
2. **Mandatory Re-Run Procedure:** When a decision rationale is flagged as void by the Compliance Audit Unit or Chief Credit Officer:
   - The original credit decision is immediately nullified.
   - The applicant's file must be scrubbed of the offending variable.
   - The underwriting evaluation must be **re-run within 48 hours** by an independent senior underwriter who did not participate in the initial assessment.
   - A formal incident report must be filed with the Fair Lending Committee within **5 business days**.

---

### 4.6 ADVERSE-ACTION NOTICE REQUIREMENTS
1. **Issuance Deadline:** If an application for credit is denied, or if terms less favorable than those requested are offered (counter-offer), a formal Adverse-Action Notice must be dispatched to the applicant via secure electronic mail and physical mail within **7 business days** of the final decision.
2. **Content Mandate:** The notice must explicitly state:
   - The principal, specific economic reasons for the adverse action (e.g., debt-to-income ratio exceeding 45%, insufficient verified credit history, or documented delinquency within the past 24 months). A generic statement of denial is legally and operationally non-compliant.
   - The name, address, and toll-free telephone number of the consumer reporting agency that provided any consumer report utilized in the decision.
   - A clear statement that the applicant has the right to request a free copy of the consumer report within **60 days** of receipt of the notice.
   - A statement detailing the applicant's right to request the specific reasons for the adverse action within **30 days** of receipt, if such reasons were not fully detailed in the initial notice.
3. **Record Retention:** A copy of every Adverse-Action Notice, along with the complete unalterable data snapshot used to generate the denial, must be retained in the Ashva Capital secure archives for a minimum of **seven (7) years** from the date of generation.
