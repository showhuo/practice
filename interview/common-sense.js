// 前端常见的面试问题

// 实现节流函数 throttle
// 基础版
function throttle(fn, timeout) {
  let waiting = false;
  return (...args) => {
    if (!waiting) {
      waiting = true;
      const start = Date.now();
      fn(...args);
      const end = Date.now(); // 函数执行时间需要算进去，否则 timeout 不准
      setTimeout(() => {
        waiting = false;
      }, timeout - (end - start));
    }
  };
}

// curry 版 throttle
// 每次点击都会被蓄能，最后全部被释放
function throttleCurried(fn, timeout) {
  let waiting = false;
  let cache = 0;
  return (...args) => {
    if (waiting) {
      // 锁定期，只蓄能，不执行
      cache += 1;
      return;
    }
    // 解锁期，先锁住窗口，全部执行
    waiting = true;
    const start = Date.now();
    while (cache > 0) {
      fn(...args);
      cache -= 1;
    }
    const end = Date.now();
    // 打开锁
    setTimeout(() => {
      waiting = false;
    }, timeout - (end - start));
  };
}

// 高阶版
// 最后一次尝试一定会被执行
function throttlePro(fn, timeout) {
  let lastTime = null,
    timer = null;
  return (...args) => {
    // 首次
    if (!lastTime) {
      fn(...args);
      lastTime = Date.now();
    } else {
      // 第二次开始，每次重新校正定时器，非常骚
      clearTimeout(timer);
      timer = setTimeout(() => {
        fn(...args);
        lastTime = Date.now();
        // 剩余时间即使小于 0 也会被浏览器重置为 4ms 左右
      }, timeout - (Date.now() - lastTime));
    }
  };
}

// 函数式编程
// 实现一个 curry 函数，它可以将普通函数 curry 化，并预置一些参数，生成新函数，接受剩余的参数，最后才执行
function curry(fn, ...preset) {
  return (...later) => fn(...preset, ...later);
}

// 实现一个 compose 函数，组装一系列标准 fn，生成终极函数，从后往前依次触发 fn
// 标准 fn：只接受一个参数，并返回一个值
const compose = (...fns) => x => fns.reduceRight((y, fn) => fn(y), x);

// 实现一个 bind 函数，跟上面的 curry 有点像，只是多了 this 指向，并且 apply 只能接收一个数组
const bind = (fn, context, ...preset) => (...args) =>
  fn.apply(context, [...preset, ...args]);

// 实现一个单例模式，注意是该实例只能被创建一个，而不是只能执行一次
// 通常是用 IIFE 闭包实现
const Singleton = (function() {
  let instance = null;
  function create(params) {
    return { params };
  }
  return {
    getInstance: params => {
      // 如果实例已创建过，直接返回同一个
      if (instance) return instance;
      instance = create(params);
      return instance;
    }
  };
})();

// 异步执行
// nextTick 坑爹 >> promise.then >> await 下一行 >> setTimeout
function asyncOrder() {
  async function a1() {
    console.log("a1 start"); // 2
    await a2();
    console.log("a1 end"); // ! 倒数第二个，在 settimeout 之前
  }
  async function a2() {
    console.log("a2"); // 3
  }

  console.log("script start"); // 1

  setImmediate(() => {
    console.log("setImmediate"); // 不稳定不建议使用，相当于 setTimeout(fn,0)
  });

  setTimeout(() => {
    console.log("setTimeout"); // last，最小间隔其实是 4ms
  }, 0);

  Promise.resolve().then(() => {
    console.log("promise1"); // 7
  });

  process.nextTick(() => console.log("next tick 1")); // 插队王

  a1();

  let promise2 = new Promise(resolve => {
    resolve("promise2.then");
    console.log("promise2"); // 5
  });

  promise2.then(res => {
    console.log(8); // 8
    Promise.resolve().then(() => {
      console.log("promise3"); // 9
    });
  });

  process.nextTick(() => console.log("next tick 2"));

  console.log("script end"); // 6
}
asyncOrder();

// 订阅发布模式
class PubSub {
  constructor(name) {
    this.name = name;
    this.listeners = {};
  }
  subScribe(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }
  publish(event, data) {
    if (!this.listeners[event] || !this.listeners[event].length) {
      return;
    }
    this.listeners[event].forEach(cb => {
      cb(data);
    });
  }
  // 取消订阅
  unSubscribe(event) {
    delete this.listeners[event];
  }
}

// !websocket 设计模式
// 给定一个 ws 对象，有 send 和 on('msg', cb) 两个方法
// 设计一个 fetch(url) 底层调用 ws，返回 promise 可以链式调用
// 需要区分不同的 ws 请求
const wsPubSuber = new PubSub();
const ws = {
  send() {},
  on(res) {
    // 收到数据应该通知对应的 id 接受者
    wsPubSuber.publish(res.id, res);
  }
};
class Fetch {
  constructor(url) {
    this.instance = ws.send(url, Fetch.id++);
    // 注册到哈希表，声明 id 与请求的关系
    return new Promise((resolve, reject) => {
      // !需要把收到的数据 resolve 出去，这里就等待 publish 触发
      wsPubSuber.subScribe(Fetch.id, res => {
        resolve(res);
      });
    });
  }
}
// !没有 babel 的帮助，类的静态属性只能写在外面，静态方法才能写在里面 static fn(){}
Fetch.id = 0;
let f1 = new Fetch(12);
let f2 = new Fetch(2);
// console.log(Fetch.id);

// CSS 高频问题
// Q1：CSS 盒模型？
// A1：默认 content-box 宽度只包含内容本身，设置 box-sizing: border-box 宽度将包含到 border
// Q2：Margin 上下合并？
// A2：1，发生在兄弟 blocks 之间；2，父元素 margin-top/bottom 与子元素之间没有 border、padding、bfc 等隔开；3，空的 block
// Q3：BFC，块状上下文？其实就是 css 的 block 概念
// A3：常见的创建 BFC 的方式有：1，overflow 不是 visible 的 block 元素；2，float、absolute、fixed 元素；3，flex 元素的直接子元素；4，inline-block；
// 5，table 相关的元素；6，display: flow-root 规范创建 BFC（Safari不支持）；
// BFC应用：1，父元素 BFC 才能完全包裹 float 子元素；2，子元素 BFC 就不会与父元素发生 Margin 重合；

// requestIdleCallback 将任务优先级降低
// 跟 debounce、throttle 概念不大一样
function throttleRIC(fn, timeout) {
  return (...args) => {
    requestIdleCallback(
      deadline => {
        // deadline 有两个属性 timeRemaining() 和 didTimeout，表示当前 frame 剩余时间是否足够
        if (deadline.timeRemaining() > 1 || deadline.didTimeout) {
          fn(...args);
        }
      },
      { timeout }
    );
  };
}

// 定时器模拟 requestIdleCallback
// 功能较弱，使用时可以自己决定剩余 0 - 50ms 的时间再执行
function mockRIC(callback) {
  const startTime = Date.now();
  return setTimeout(() => {
    callback({
      timeRemaining() {
        return Math.max(0, 50.0 - (Date.now() - startTime));
      }
    });
  }, 1);
}

// 单线程切片计算，每次最多占用 10 ms，然后让出线程
function sliceAndCal(arr, interval = 10) {
  const now = Date.now();
  while (Date.now() - now <= interval) {
    const todo = arr.shift();
    // doSomeThing with todo
  }
  setTimeout(() => {
    sliceAndCal(arr, interval);
  }, 0);
}
