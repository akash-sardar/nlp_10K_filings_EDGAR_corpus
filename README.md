# TASK_01
## Coding Assignment Task # 1
- Dataset: IEDGAR SEC filings (public data) -
- https://huggingface.co/datasets/eloukas/edgar-corpus
- Language: Python
- Implementation: Pyspark
- Submission: Github repository containing code + plots (jpeg).
- Expected Maximum Duration: 3 hours

Given a set of documents: create a solution that allows the end user to understand the documents in a two dimensional space and to identify outliers.
* Dataset:
- Year: 2020
- Filing type: 10K
- Sections: All
- Companies: Limit to 10.

* Steps:
1. Convert the documents to chunks,
2. Convert the chunks into embeddings,
3. Standard scale the embeddings,
4. Perform principal components analysis,
5. Apply dimensionality reduction,
6. Perform Kmeans clustering and assign chunks an cluster number.
7. Create an outlier flag.
8. Plot(s)
- Embeddings in 2 dimensions
- Colored by assigned clusters.
- Colored by outlier flag
- Colored by section number.

# Dataset card:
This dataset card is based on the paper EDGAR-CORPUS: Billions of Tokens Make The World Go Round authored by Lefteris Loukas et.al, as published in the ECONLP 2021 workshop.
This dataset contains the annual reports of public companies from 1993-2020 from SEC EDGAR filings.
There is supported functionality to load a specific year.
Care: since this is a corpus dataset, different train/val/test splits do not have any special meaning. It's the default HF card format to have train/val/test splits.
If you wish to load specific year(s) of specific companies, you probably want to use the open-source software which generated this dataset, EDGAR-CRAWLER: https://github.com/nlpaueb/edgar-crawler.

### Below is the metadata definition

{
  "cik": "320193",
  "company": "Apple Inc.",
  "filing_type": "10-K",
  "filing_date": "2022-10-28",
  "period_of_report": "2022-09-24",
  "sic": "3571",
  "state_of_inc": "CA",
  "state_location": "CA",
  "fiscal_year_end": "0924",
  "filing_html_index": "https://www.sec.gov/Archives/edgar/data/320193/0000320193-22-000108-index.html",
  "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm",
  "complete_text_filing_link": "https://www.sec.gov/Archives/edgar/data/320193/0000320193-22-000108.txt",
  "filename": "320193_10K_2022_0000320193-22-000108.htm",
  "item_1": "Item 1. Business\nCompany Background\nThe Company designs, manufactures ...",
  "item_1A": "Item 1A. Risk Factors\nThe Company’s business, reputation, results of ...",
  "item_1B": "Item 1B. Unresolved Staff Comments\nNone.",
  "item_1C": "",
  "item_2": "Item 2. Properties\nThe Company’s headquarters are located in Cupertino, California. ...",
  "item_3": "Item 3. Legal Proceedings\nEpic Games\nEpic Games, Inc. (“Epic”) filed a lawsuit ...",
  "item_4": "Item 4. Mine Safety Disclosures\nNot applicable. ...",
  "item_5": "Item 5. Market for Registrant’s Common Equity, Related Stockholder ...",
  "item_6": "Item 6. [Reserved]\nApple Inc. | 2022 Form 10-K | 19",
  "item_7": "Item 7. Management’s Discussion and Analysis of Financial Condition ...",
  "item_8": "Item 8. Financial Statements and Supplementary Data\nAll financial ...",
  "item_9": "Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure\nNone.",
  "item_9A": "Item 9A. Controls and Procedures\nEvaluation of Disclosure Controls and ...",
  "item_9B": "Item 9B. Other Information\nRule 10b5-1 Trading Plans\nDuring the three months ...",
  "item_9C": "Item 9C. Disclosure Regarding Foreign Jurisdictions that Prevent Inspections\nNot applicable. ...",
  "item_10": "Item 10. Directors, Executive Officers and Corporate Governance\nThe information required ...",
  "item_11": "Item 11. Executive Compensation\nThe information required by this Item will be included ...",
  "item_12": "Item 12. Security Ownership of Certain Beneficial Owners and Management and ...",
  "item_13": "Item 13. Certain Relationships and Related Transactions, and Director Independence ...",
  "item_14": "Item 14. Principal Accountant Fees and Services\nThe information required ...",
  "item_15": "Item 15. Exhibit and Financial Statement Schedules\n(a)Documents filed as part ...",
  "item_16": "Item 16. Form 10-K Summary\nNone.\nApple Inc. | 2022 Form 10-K | 57"
}