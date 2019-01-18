import matplotlib.pyplot as plt
import re



text = """Parameter containing:
tensor([[0.1399]], requires_grad=True)
Parameter containing:
tensor([[0.0029]], requires_grad=True)
Parameter containing:
tensor([[-0.0573]], requires_grad=True)
Parameter containing:
tensor([[0.0080]], requires_grad=True)
Parameter containing:
tensor([[0.0171]], requires_grad=True)
Parameter containing:
tensor([[0.0504]], requires_grad=True)
Parameter containing:
tensor([[0.0731]], requires_grad=True)
Parameter containing:
tensor([[0.0806]], requires_grad=True)
Parameter containing:
tensor([[-0.0041]], requires_grad=True)
Parameter containing:
tensor([[0.0057]], requires_grad=True)
"""

dim = 1

data = []

pattern = re.compile(r"[-+]?\d*\.\d+|\d+") 

for line in text.split("\n"):
	# result = pattern.match(line)
	result = re.findall(r"[-+]?\d*\.\d+|\d+", line)

	# print(line)
	# print(result)
	if result:
		print(result)
