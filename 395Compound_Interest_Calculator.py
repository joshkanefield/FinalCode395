

#Simple Compound Interest Calculator

def compound_interest():
	principal = float(input('Enter the starting principal: '))
	interest_rate = float(input('Enter the annual interest rate: '))
	compound_rate = float(input('How many times per year is the interest compounded?: '))
	time_year = float(input('For how many years will the account earn interest?: '))

	Amount = principal * (1 + ((interest_rate/100) / compound_rate)) ** (compound_rate * time_year)

	print('At the end of', (time_year) , 'years you will have $ ', Amount)