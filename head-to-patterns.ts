interface Flyable {
    fly: () => void;
}

interface Quackable {
    quack: () => void;
}

// 把容易变化的行为，抽象成类，目的是让它原子化，方便后面自由组合以及修改
class NormalFly implements Flyable {
    fly() {
        console.log(`I fly like normal!`);
    }
}

class RocketFly implements Flyable {
    fly() {
        console.log('I fly like rocket!');
    }
}

class SilentQuack implements Quackable {
    quack() {
        console.log('I quack  silently!');
    }
}

class NormalQuack implements Quackable {
    quack() {
        console.log('I quack like normal!');
    }
}

class Duck {
    name: string
    flyBehavior: Flyable
    quackBehavior: Quackable
    constructor(name: string) {
        this.name = name;
    }
    performFly() {
        this.flyBehavior.fly();
    }
    performQuack() {
        this.quackBehavior.quack();
    }
    displayName() {
        console.log(this.name);
    }
    // 增加 setter 以支持动态变化
    setFlyBehavior(fb: Flyable) {
        this.flyBehavior = fb;
    }
    setQuackBehavior(qb: Quackable) {
        this.quackBehavior = qb;
    }
}

// 我们希望 fly 行为的改变尽可能灵活和独立，又能复用其他代码
// 这里的火箭 🦆 除了 fly 的行为不大一样，其他跟普通鸭子一模一样
class RocketDuck extends Duck {
    flyBehavior = new RocketFly();
}

function getDuck() {
    const randomIdx = Math.round(Math.random());
    return [new RocketDuck('R'), new Duck('D')][randomIdx];
}

// 假设我们需要随机选一只鸭子让他表演飞行，就可以不关心它是什么 🦆，只知道它会飞
const duck: Duck = getDuck();
duck.performFly();
// 触发策略模式，可以自由替换一组 fly 行为
duck.setFlyBehavior(new NormalFly());
duck.performFly();


// 观察者模式
interface Subject {
    registerObserver: (ob: Observer) => void;
    unRegisterObserver: (ob: Observer) => void;
    notifyObservers: () => void;
}

interface Observer {
    update: (num: number) => void;
}

class ConcreteSubject implements Subject {
    observers: Set<Observer> = new Set()
    someData: number
    registerObserver(ob: Observer) {
        this.observers.add(ob);
    }
    unRegisterObserver(ob: Observer) {
        this.observers.delete(ob);
    }
    notifyObservers() {
        // 这里默认是 push 模式，如果是 pull 模式就不在这里传数据，而是让 observer 在 update 自己去取
        this.observers.forEach(ob => { ob.update(this.someData) })
    }
    changeData() {
        this.someData++;
    }
    getData() {
        return this.someData;
    }
}

class ConcreteObserver implements Observer {
    name: string
    constructor(name: string, subject: Subject) {
        this.name = name;
        subject.registerObserver(this);
    }
    update(num: number) {
        // 这里默认是 push 模式，如果是 pull 模式就在这里主动去调用 observable 的 getData
        console.log(`Subject' data changed to ${num}, ${this.name} needs to update!`);
    }
}


// Decorator 模式
// 被装饰过的对象，拥有与之前一致的接口，因此要求 Decorator 类继承 component 父类
// 在 JS 里通常不是装饰器类，而是 @decorator 函数
abstract class Beverage {
    name: string;
    size: string;
    getName() {
        return this.name;
    }
    getSize() {
        return this.size;
    }
    setSize(size: string) {
        this.size = size;
    }
    abstract cost(): number;
}

class Coffee extends Beverage {
    name = 'coffee';
    prices = { small: 1.0, middle: 1.5, large: 2.0 }
    cost() {
        return this.prices[this.size]
    }
}

abstract class BeverageDecorater extends Beverage {
    abstract getName(): string
}

class MochaDecorater extends BeverageDecorater {
    beverage: Beverage;
    prices = { small: 0.1, middle: 0.2, large: 0.3 }
    constructor(beverage: Beverage) {
        super();
        this.beverage = beverage;
    }
    getName() {
        return this.beverage.getName() + ', ' + 'mocha';
    }
    cost() {
        return this.beverage.cost() + this.prices[this.size];
    }
}

const middleCupCoffee = new Coffee();
middleCupCoffee.setSize('middle');
const mochaCoffee = new MochaDecorater(middleCupCoffee);
const moreMocha = new MochaDecorater(mochaCoffee);


// 工厂模式：工厂方法和抽象工厂模式，前者生产一个对象，并定义其他方法操作，用继承实现；后者生产一系列对象，用接口实现；
// 封装创建对象的过程，让使用者与创建者解耦
abstract class FactoryMethodFactory {
    abstract createProduct(): Product
    deliver() {
        // deliver product to customers，whatever product it is.
        console.log(`Deliver ${this.createProduct()} to customers`);
    }
}

class Product {
    name: string;
    price: number;
    constructor(name: string, price: number) {
        this.name = name;
        this.price = price;
    }
}

class ConcreteFactoryMethodFactory extends FactoryMethodFactory {
    createProduct() {
        return new Product('A', 1.0)
    }
}

// 抽象工厂用接口，需要提前实现所有可能的具体工厂，让 client 自由选择
interface AbstractFactory {
    createProductA(): Product;
    createProductB(): Product;
    createProductC(): Product;
}


// 单例模式，用静态方法和静态属性实现
// 因为 JS 是单线程，不需要给 getInstance 上锁
class Singleton {
    static uniqueInstance: Singleton;
    static getInstance() {
        if (!this.uniqueInstance) {
            this.uniqueInstance = new Singleton();
        } else {
            return this.uniqueInstance;
        }
    }
    // 无法被外界实例化，只能通过 getInstance 获取单例
    private constructor() { }

    otherMethod() { }
}


// 命令模式，三种角色：command、receiver、Invoker，命令对象需要先注册到 Invoker，再被后者使用
interface Command {
    execute(): void;
    undo(): void;
}

class Receiver {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
    someActions() {
        console.log(`${this.name} do some actions`);
    }
}

class Invoker {
    // 可以支持多个，比如一组按键对应不同命令
    myCommand: Command;
    setCommand(command: Command) {
        this.myCommand = command;
    }
    executeMyCommand() {
        this.myCommand.execute();
    }
    undoMyCommand() {
        this.myCommand.undo();
    }
}

class SimpleCommand implements Command {
    receiver: Receiver;
    constructor(receiver: Receiver) {
        this.receiver = receiver;
    }
    execute() {
        this.receiver.someActions();
    }
    undo() { }
}

// 宏命令，自身也是一个命令对象，它可以执行一系列参数命令
class MacroCommand implements Command {
    commands: Command[];
    constructor(commands: Command[]) {
        this.commands = commands;
    }
    execute() {
        this.commands.forEach(c => { c.execute() })
    }
    undo() {
        this.commands.forEach(c => { c.undo() })
    }
}

const invoker = new Invoker();
const receiverA = new Receiver('A');
const receiverB = new Receiver('B');
const simpleCommandA = new SimpleCommand(receiverA);
const simpleCommandB = new SimpleCommand(receiverB);
const macroCommand = new MacroCommand([simpleCommandA, simpleCommandB]);
invoker.setCommand(simpleCommandA);
invoker.executeMyCommand();
invoker.setCommand(macroCommand);
invoker.executeMyCommand();
invoker.undoMyCommand();


// 适配器模式将不适配的接口们转换成一个适配 Client 的接口
// 表面模式（Facade）简化接口，方便使用
interface DuckType {
    quack(): void;
    fly(): void;
}

class Turkey {
    goggle() {
        console.log(`I am a turkey goggling`);
    }
    fly() {
        console.log(`Turkey flying!`);
    }
}

class TurkeyToDuckAdapter implements DuckType {
    turKey: Turkey;
    constructor(turkey: Turkey) {
        this.turKey = turkey;
    }
    quack() {
        this.turKey.goggle();
    }
    fly() {
        this.turKey.fly();
    }
}

class DuckA implements DuckType {
    quack() {
        console.log(`Duck quacking`);
    }
    fly() {
        console.log(`Duck flying`);
    }
}

class FacadeForAnimals {
    duck: DuckType;
    turkey: Turkey;
    constructor(duck: DuckType, turkey: Turkey) {
        this.duck = duck;
        this.turkey = turkey;
    }
    makeNoise() {
        this.duck.quack();
        this.turkey.goggle();
    }
}


// 模板方法模式 Template Method
abstract class TemplateClass {
    templateMethod() {
        this.lifeCycleMethodA();
        this.lifeCycleMethodB();
        this.lifeCycleMethodC();
        if (this.hookD()) {
            this.hookD();
        }
    }
    // 这两个必须由子类实现
    abstract lifeCycleMethodA(): void;
    abstract lifeCycleMethodB(): void;
    lifeCycleMethodC() {
        console.log(`lifeCycleMethodC is shared by all.`);
    }
    hookD() {
        console.log(`hookD is a hook method, it is optional for subclasses`);
        return null;
    }
}


// 组合模式，composite 节点和 leaf 节点拥有相同接口，Client 不需要区分这两种节点
abstract class ComponentNode {
    add() {
        throw new Error("No add method available");
    }
    rename() {
        throw new Error("No add method available");
    }
    print() {
        throw new Error("No add method available");
    }
}

class CompositeNode extends ComponentNode {
    add() {
        console.log(`CompositeNode is adding something`);
    }
    print() {
        console.log(`CompositeNode is printing`);
    }
}

class LeafNode extends ComponentNode {
    rename() {
        console.log(`LeafNode is renaming`);
    }
    print() {
        console.log(`LeafNode is printing`);
    }
}

class ClientForCompositePattern {
    node: ComponentNode;
    constructor(node: ComponentNode) {
        this.node = node;
    }
    addNode() {
        this.node.add();
    }
    renameNode() {
        this.node.rename();
    }
    printNode() {
        this.node.print();
    }
}


// State pattern 状态模式，又称为有限状态机，有点像策略模式，但针对性不同
// 所有的状态类实现相同的 State 接口，尽管它们的 actions 名称相同，但作用不相同
// Machine 类将 actions 调用委托给 State state，调用之后 state 可能会发生替换，它们相互影响
abstract class StatePatternState {
    abstract name: string;
    insertCoin() {
        console.error(`can not insert coin at state ${this.name}`);
    }
    play() {
        console.error(`can not play at state ${this.name}`);
    }
    ejectCoin() {
        console.error(`can not eject coin at state ${this.name}`);
    }
}

class StatePatternMachine implements StatePatternState {
    name: 'Finite state machine';
    noCoinState: NoCoinState;
    hasCoinState: HasCoinState;
    isBrokenState;
    currentState: StatePatternState;
    coins: number = 0;
    constructor(noCoinState: NoCoinState, hasCoinState) {
        this.noCoinState = noCoinState;
        this.hasCoinState = hasCoinState;
    }
    insertCoin() {
        this.currentState.insertCoin();
    }
    play() {
        this.currentState.play();
    }
    ejectCoin() {
        this.currentState.ejectCoin();
    }
    setState(state: StatePatternState) {
        this.currentState = state;
    }
    addCoin() {
        this.coins++;
    }
    reduceCoin() {
        if (this.coins > 0) this.coins--;
    }
}

class NoCoinState extends StatePatternState {
    name: 'NoCoinState';
    machine: StatePatternMachine
    constructor(machine: StatePatternMachine) {
        super()
        this.machine = machine;
    }
    insertCoin() {
        this.machine.addCoin();
        this.machine.setState(this.machine.hasCoinState);
        console.log('Now you can play!');
    }
    play() {
        console.error('Pls insert coin first!');
    }
}

class HasCoinState extends StatePatternState {
    name: 'HasCoinState';
    machine: StatePatternMachine
    constructor(machine: StatePatternMachine) {
        super()
        this.machine = machine;
    }
    insertCoin() {
        this.machine.addCoin();
        console.log(`Now you have ${this.machine.coins} coins, you can play!`);
    }
    play() {
        this.machine.reduceCoin();
        console.log(`Game start!`);
    }
    ejectCoin() {
        // 思考题：coins 数量判断由 Machine 负责，还是 State 自己负责？
        // 在实践中我们常说，状态应该尽可能由上层管理，下层组件应该是纯函数
        // 又说组件应该内聚，它们是矛盾的吗？
    }
}


// Proxy pattern 代理模式，就是访问控制
// 比如：权限控制、虚拟代理、缓存代理等等
interface ProxyPatternSubject {
    methodNeedsAuth(): void;
    methodBNeedsTimeToCreate(): void;
    methodCPreferCache(): object;
}

class ProxyPatternProxy implements ProxyPatternSubject {
    subject: ProxyPatternSubject;
    hasAuth: boolean;
    lazyObject: ProxyPatternSubject;
    virtualObject: ProxyPatternSubject;
    cacheObject: object;
    constructor(subject: ProxyPatternSubject) {
        this.subject = subject;
    }
    methodNeedsAuth() {
        if (this.hasAuth) {
            this.subject.methodNeedsAuth();
        } else {
            console.error(`${this.subject} has no permission`);
        }
    }
    methodBNeedsTimeToCreate() {
        if (this.lazyObject) {
            // 已创建，可使用目标
            this.lazyObject.methodBNeedsTimeToCreate();
        } else {
            // 创建开销较大，比如网络请求，此时先使用虚拟对象，假设5秒创建完成，那时再使用真实对象
            // 不同于缓存代理，假设业务需要每次真实创建
            this.virtualObject.methodBNeedsTimeToCreate();
            setTimeout(() => {
                this.lazyObject.methodBNeedsTimeToCreate();
            }, 5000);
        }
    }
    methodCPreferCache() {
        if (this.cacheObject) {
            return this.cacheObject;
        } else {
            // 同步创建或者异步创建都有可能，根据业务需要
            // 如果是异步创建，无论是否有缓存，都包装成 Promise 给 Client 使用
        }
    }
}