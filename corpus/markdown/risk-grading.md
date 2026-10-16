# Internal Risk Grading and Pricing Grid

# ASHVA CAPITAL INTERNAL CREDIT POLICY MANUAL
**Document Reference:** CP-M-882-RiskGrading  
**Effective Date:** 14 January 2025  
**Owner:** Office of the Chief Credit Officer  
**Applies To:** All Underwriting Staff, Automated Decision Engines, and Credit Reviewers  

---

## 4. RISK GRADING METHODOLOGY AND PRICING MATRIX

### 4.1 General Rule
4.1.1 The underwriting system shall assign an internal risk grade to each commercial credit application based strictly on the quantitative criteria set forth in Section 4.2.  
4.1.2 An application shall be allocated the highest available risk grade for which it simultaneously satisfies both the minimum Debt Service Coverage Ratio (DSCR) and the minimum bureau score thresholds.  
4.1.3 The lending base rate for Ashva Capital is fixed at 9.50 percent per annum. The indicative interest rate for any approved facility shall be the sum of the base rate (9.50 percent) and the applicable spread defined for the assigned grade.  

### 4.2 Risk Grading Grid
The eight internal risk grades, along with their quantitative thresholds, indicative spreads, maximum tenors, and collateral coverage requirements, are strictly defined as follows:

1. **Grade A1**  
   * **DSCR Criteria:** At least 2.00  
   * **Bureau Score Criteria:** At least 750  
   * **Interest Rate Spread:** 150 basis points over the base rate (Effective Rate: 11.00 percent)  
   * **Maximum Tenor Permitted:** 84 months  
   * **Collateral Coverage Required:** 0 percent of the loan amount  

2. **Grade A2**  
   * **DSCR Criteria:** At least 1.75  
   * **Bureau Score Criteria:** At least 700  
   * **Interest Rate Spread:** 200 basis points over the base rate (Effective Rate: 11.50 percent)  
   * **Maximum Tenor Permitted:** 72 months  
   * **Collateral Coverage Required:** 25 percent of the loan amount  

3. **Grade B1**  
   * **DSCR Criteria:** At least 1.50  
   * **Bureau Score Criteria:** At least 650  
   * **Interest Rate Spread:** 275 basis points over the base rate (Effective Rate: 12.25 percent)  
   * **Maximum Tenor Permitted:** 60 months  
   * **Collateral Coverage Required:** 50 percent of the loan amount  

4. **Grade B2**  
   * **DSCR Criteria:** At least 1.35  
   * **Bureau Score Criteria:** At least 600  
   * **Interest Rate Spread:** 350 basis points over the base rate (Effective Rate: 13.00 percent)  
   * **Maximum Tenor Permitted:** 60 months  
   * **Collateral Coverage Required:** 75 percent of the loan amount  

5. **Grade C1**  
   * **DSCR Criteria:** At least 1.25  
   * **Bureau Score Criteria:** At least 550  
   * **Interest Rate Spread:** 450 basis points over the base rate (Effective Rate: 14.00 percent)  
   * **Maximum Tenor Permitted:** 48 months  
   * **Collateral Coverage Required:** 100 percent of the loan amount  

6. **Grade C2**  
   * **DSCR Criteria:** At least 1.15  
   * **Bureau Score Criteria:** At least 500  
   * **Interest Rate Spread:** 575 basis points over the base rate (Effective Rate: 15.25 percent)  
   * **Maximum Tenor Permitted:** 36 months  
   * **Collateral Coverage Required:** 125 percent of the loan amount  

7. **Grade D1**  
   * **DSCR Criteria:** At least 1.00  
   * **Bureau Score Criteria:** At least 450  
   * **Interest Rate Spread:** 725 basis points over the base rate (Effective Rate: 16.75 percent)  
   * **Maximum Tenor Permitted:** 24 months  
   * **Collateral Coverage Required:** 150 percent of the loan amount  

8. **Grade D2**  
   * **DSCR Criteria:** Any application failing to meet the minimum criteria of Grade D1  
   * **Bureau Score Criteria:** Any application failing to meet the minimum criteria of Grade D1  
   * **Interest Rate Spread:** 900 basis points over the base rate (Effective Rate: 18.50 percent)  
   * **Maximum Tenor Permitted:** 12 months  
   * **Collateral Coverage Required:** 150 percent of the loan amount  

### 4.3 Tenor Constraints and Exceptions
4.3.1 The final tenor offered to a borrower shall not exceed the maximum tenor permitted by the assigned risk grade or the maximum tenor permitted by the applicable sector policy, whichever is shorter.  
4.3.2 If a borrower requests a tenor that exceeds either the grade limit or the sector limit, the underwriting system shall flag the application with reason code `TENOR_ABOVE_LIMIT` and route the file directly to a human credit reviewer. Auto-approval is strictly prohibited under this condition.  

### 4.4 Collateral Valuation and Shortfalls
4.4.1 Collateral coverage is calculated as a strict percentage of the total requested loan amount based on verified independent valuation.  
4.4.2 If the appraised value of the pledged collateral is less than the mandatory coverage percentage stipulated for the assigned risk grade in Section 4.2, the underwriting system shall flag the application with reason code `COLLATERAL_SHORTFALL`.  
4.4.3 Any application tagged with `COLLATERAL_SHORTFALL` shall be stripped of auto-approval status and transferred to a human reviewer for manual credit exception processing.  

### 4.5 Auto-Approval Parameters and Restrictions
4.5.1 Applications assigned risk grades **A1, A2, B1, B2, and C1** are eligible for automated decisioning and auto-approval, provided the requested loan amount does not exceed **1,000,000 rupees**.  
4.5.2 Applications assigned risk grades **C2, D1, and D2** are strictly prohibited from receiving auto-approval at any loan size, regardless of the requested amount. All such applications must undergo mandatory manual review by a designated underwriter.
