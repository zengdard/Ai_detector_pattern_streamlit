import requests
#tokenizer = AutoTokenizer.from_pretrained("username/repo_name")
#model = AutoModel.from_pretrained("username/repo_name")


token = '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'
API_URL = "https://api-inference.huggingface.co/models/roberta-large-openai-detector"
headers = {"Authorization": f"Bearer {token}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I like you. I love you",
})
print(output)