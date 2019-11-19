// 前端常见的编程问题

// 实现节流函数 throttle
// 简易版
function throttle(fn, timeout) {
  let waiting = false;
  return (...args) => {
    if (!waiting) {
      waiting = true;
      fn(...args);
      setTimeout(() => {
        waiting = false;
      }, timeout);
    }
  };
}

// 增强版
// 确保最后一下点击一定会被执行
function throttlePro(fn, timeout) {
  let lastTime = null,
    timer = null;
  return (...args) => {
    if (!lastTime) {
      fn(...args);
      lastTime = Date.now();
    } else {
      clearTimeout(timer);
      timer = setTimeout(() => {
        fn(...args);
        lastTime = Date.now();
        // 剩余时间即使小于 0 也会被浏览器重置为 4ms 左右
      }, timeout - (Date.now() - lastTime));
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
    while (cache > 0) {
      fn(...args);
      cache -= 1;
    }
    // 打开锁
    setTimeout(() => {
      waiting = false;
    }, timeout);
  };
}

// 函数式编程
// 实现一个 curry 函数，它可以将普通函数 curry 化，先接受一些参数，然后返回新函数接受剩余的参数，最后才执行
function curry(fn, ...preset) {
  return (...later) => fn(...preset, ...later);
}

// 实现一个 compose 函数，接收一系列 fn 返回一个函数，该函数接受一个参数，然后从右到左依次调用 fn，将中间结果传递下去，最后返回结果
const compose = (...fns) => x => fns.reduceRight((y, fn) => fn(y), x);

// 实现一个 bind 函数，跟上面的 curry 有点像，只是多了 this 指向，并且 apply 只能接收一个数组
function bind(fn, context, ...preset) {
  return (...args) => fn.apply(context, [...preset, ...args]);
}

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

// 这个函数把 fn 转化为只能执行一次的函数，其实也算单例的概念
function singletonFactory(fn) {
  let lock = false;
  return (...args) => {
    if (lock) return;
    lock = true;
    return fn(...args);
  };
}
