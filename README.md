Using  tf-idf  and  cosine  similarity  into  Shazam-based  fingerprinting algorithms
===========

This  study  demonstrates  the  application  of  text retrieval techniques into Shazam-like fingerprinting algorithms. The goal is to filter query input hashes using tf-idf as a measure
of relevance in order to reduce the number of records returned by  the  inverted index. As  accuracy  could  potentially  be  affected  bythis filtering approach, we investigate these requisites together, by  looking  for  a  filtering  threshold τ that  produces  reduced database response payloads with minimal impact on accuracy.

This experiment also  discusses  the  use  of  cosine  similarity  as  analternative  to  the  original  Shazam's  scoring  method,  giventhat  the  last  one  restricts  the  algorithm's  robustness  against time  distortions.  We  demonstrate  that  the  application  of  these techniques outperforms the original algorithms description for many different datasets.

