Subject: Re: CoD2M2_datacollection

Hi Peiling,

Yes, the Google Fact Check Explorer looks usable for a pilot, and it seems much more practical than the older X/Twitter-based datasets.

I checked the recent-results page and the data appears to cover a range of topics rather than a single narrow category. In a small sample, I found examples related to elections/politics, health claims, AI-generated or manipulated images, transport/public-policy claims, and consumer/brand misinformation.

Google also provides two useful routes for data collection:

1. Fact Check Tools API
- This appears to expose the same fact-check result pool as the Explorer.
- Useful metadata includes claim text, claimant, claim date, review URL, publisher, review date, textual rating, and language.

2. Data Commons / ClaimReview research dataset and daily feed
- Google provides a historical research dataset of ClaimReview markup, plus a public daily feed.
- These contain structured fact-check metadata and links back to the original publisher pages.
- One important limitation is that the full body text of the fact-check article is not included in the dataset itself, so if we need article text later, we may need a second collection pass from publisher URLs.

I have started preparing a small starter sample and a normalization script so we can test data integrity and usability before scaling up.

Best,
Michael
