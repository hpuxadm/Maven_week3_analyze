**Same disclaimer as week #2
**I'm still learning Python so had to lean heavily on GPT to produce the actual code.
---


This is the work for Week 3 which would include a document analysis python script that would potentially fall under the "Software/Code 3.0" characterization.

The premise behind the script is to prompt the user for a directory local to their machine. If the directory is not found, then it will be created. If the directory is found, the script will then can said directory for files that qualify for GPT-mini analysis(.pdf, .docx, etc).

The relevant files in said directory are sent to the console and the user can then select and specify which file should be analyuzed.

Upon successful selection, the script then scans the file in order to understand the amount of tokens that are contained in that file. This is done to provide a token/stream in and token/stream out estimate, so that a cost projection for the analysis can be provided to the console.

Upon running, we currently have the script dissecting or "chunking" the file into small segment sizes. This was initially done as we were running into rate limit issues on the free tier of gpt-4o-mini.

We found that by chunking and lmiiting the token sizes of those chunks, we could effectively complete the job, however, due to the limitations of the free-tier, input was throttled and runs of documentation analysis like the Constitution of the United States as an example, were taking more than an hour. We finally upgraded to the pay as you go model, just to quantify the performance and the performance drastically improved(of course). Considering the amount of tokens being ingressed and egressed thru the API - I don't think I have even approached close to 10 cents in total cost for the multiple runs, test, and failures that I had initiated and experienced.

As the run is running, I felt it would be adventageous to include a percentage complete and time to completion estimate which dynamically updates as the file is being chunked and processed. Finally, one the file has completed it will output the analysis to the console and provide a document summary in the directory that is easily identifiable via a prefix that is added to every file name vi (e.g. AI_Analysis_constitution.pdf.doc)

Things I would have like to have included are:

Batch processing of multiple files that were dropped into the repository directory It would have been interesting to test images like .gifs, .jpegs, scans, etc. Optimizations - I really began to get curious about what I could do via streaming, async IO, variable tweaking via chunk sizing, timeout reconfigurations(via API connectivity), etc, but simply ran out of time.

**ALSO NOTE: I completely understand that the way in which I placed API key assignment statically in my script is a security risk and not a best practice.  
For reasons for which I don't understand, I had issues initiating API_KEY assignment from the .env or even assigning the key variables via the terminal.  For whatever reason I could not get this functionality
to work and had to go with the static assignment in the script, in order to examine and test functionality.

