A design pattern is neither a static solution nor an algorithm. It is a repeatable software design approach to a common design problem.

========================== Creational Patterns =============================
deal with object creation

---------------------------- Builder Pattern -----------------------------
--- we do not want to have a complex constructor memeber or one that needs many arguments
--- simplifies the building process, make it more readable
--- separate the construction of a complex object from its representation so that the same construction process can create different objects representations
--- would be much easier to read when walking through a large procedural method
--- Requires creating a separate ConcreteBuilder for each different type of Product.
--- don't need too many constructors within the product class

class Pizza{
	public:
		//setters
		....
		void open() const{
			.....
		}
	private:
		string m_dough;
		string m_sauce;
		string m_topping;
};

class Cook{
	void makePizza(){...}
	void openPizza(){...}
};

// there are many different types of pizzas with different parameters 
// SpicyPizza: "cross", "mild", "ham+pineapple"
// HawaiianPizza: "pan baked", "hot", "pepperoni+salami"
// ....

// it would be messy if we use the constructor in Pizza to build those different pizza types
// use builder instead

class PizzaBuilder{
	virtual ~PizzaBuilder(){};
	virtual setters.....
};

class SpicyPizzaBuilder : public PizzaBuilder{
	......
};

class HawaiianPizzaBuilder : public PizzaBuilder{
	......
};

int main(){ // very clean, no messy constructors
// this is even so in more complicated object creation

	Cook cook;
	HawaiianPizzaBuilder hawaiianPizzaBuilder;
	SpicyPizzaBuilder    spicyPizzaBuilder;

	cook.makePizza(&hawaiianPizzaBuilder);
	cook.openPizza();

	cook.makePizza(&spicyPizzaBuilder);
	cook.openPizza();
}


-------------------------------------- Factory Pattern --------------------------------
--- factory pattern: too many types of similar objects, avoid too many "if"
--- concrete instantiation is done inside the factory class/object
--- avoid redundant code for multiple users
--- easy to modify, only have to add a type in the factory instead of everywhere 
