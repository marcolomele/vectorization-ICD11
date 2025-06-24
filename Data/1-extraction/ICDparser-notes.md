Notes about [ICDparser.ipynb](ICDparser.ipynb).

# Accessing data with API
Output: list of JSONs objects.
## Access token function
Authenticates with WHO API by sending a payload with credetials, making a POST request to get an access token, and returning the access token for subsequent API calls.

POST request – HTTP request method that follows the OAuth 2.0 specification. The response is a JSON object with an access token. 

OAuth 2.0 – open standard for authorization. It allows users to grant limited access to their resources on one site to another site, without having to expose their credentials.

## Fetch entity function
Depth-first search on the tree with recursion. Base case if entity has already been visited, process node, extract children of node, iterate through children by recursively calling the function.

Limit numnber of children to explore for testing development purposes.

Optimise. 

## Crawler
API headers – metadata about the request following REST API conventions and OAuth 2.0 Bearer token authentication scheme. 

Dump the root ids as a checkpoint for interrupted crawls; can do separate crawls for different root ids. 

# Converting to df
