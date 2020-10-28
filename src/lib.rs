use std::cell::RefCell;
use std::rc::Rc;

// Add Digits
pub fn add_digits(num: i32) -> i32 {
    let mut tmp = num;
    loop {
        if tmp < 10 {
            return tmp;
        }
        let mut tmp2 = 0;
        loop {
            if tmp == 0 {
                break;
            }
            let m = tmp % 10;
            tmp2 += m;
            tmp /= 10;
        }
        tmp = tmp2;
    }
}

// Total Hamming Distance
pub fn total_hamming_distance(nums: Vec<i32>) -> i32 {
    let (mut i, mut res) = (0, 0);
    while i < nums.len() {
        let mut j = i + 1;
        while j < nums.len() {
            res += (nums[i] ^ nums[j]).count_ones() as i32;
            j += 1;
        }
        i += 1;
    }
    res
}

// Minimum Moves to Equal Array Elements II
pub fn min_moves2(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let m = nums[nums.len() / 2];
    nums.iter().map(|x| (x - m).abs()).sum()
}

// Repeated Substring Pattern
pub fn repeated_substring_pattern(s: String) -> bool {
    for i in 1..(s.len() / 2 + 1) {
        if s.len() % i != 0 {
            continue;
        }
        let subs = &s[..i];
        if s.eq(&subs.repeat(s.len() / i)) {
            return true;
        }
    }
    false
}

// Island Perimeter
pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            match grid[i][j] {
                1 => {
                    res += 2;
                    if i > 0 {
                        res -= grid[i - 1][j];
                    }
                    if j > 0 {
                        res -= grid[i][j - 1];
                    }
                }
                _ => (),
            }
        }
    }
    res * 2
}

// Target Sum
pub fn find_target_sum_ways(nums: Vec<i32>, s: i32) -> i32 {
    fn find(nums: &Vec<i32>, s: i32, i: usize, t: i32, r: &mut i32) {
        if i >= nums.len() {
            if t == s {
                *r += 1;
            }
            return;
        }
        find(nums, s, i + 1, t + nums[i], r);
        find(nums, s, i + 1, t + (-nums[i]), r);
    }
    let r = &mut 0i32;
    find(&nums, s, 0, 0, r);
    *r
}

pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
    let (mut m, mut res) = (std::collections::HashMap::new(), vec![]);

    for v in nums.iter() {
        if let Some(_) = m.get(v) {
            res.push(*v);
        } else {
            m.insert(v, 1);
        }
    }

    res
}

// Reverse Vowels of a String
pub fn reverse_vowels(s: String) -> String {
    fn is_vowel(c: char) -> bool {
        match c {
            'a' | 'e' | 'i' | 'o' | 'u' => true,
            _ => false,
        }
    }

    let (mut i, mut j) = (0, s.len().checked_sub(1).unwrap_or(0));
    let mut cs = s.chars().collect::<Vec<char>>();

    while i < j {
        if !is_vowel(cs[i].to_ascii_lowercase()) {
            i += 1;
            continue;
        }

        if !is_vowel(cs[j].to_ascii_lowercase()) {
            j -= 1;
            continue;
        }

        cs.swap(i, j);
        i += 1;
        j -= 1;
    }
    cs.iter().collect()
}

// Missing Number
pub fn missing_number(nums: Vec<i32>) -> i32 {
    let mut s1 = nums.len();
    let mut s2 = 0;
    for (i, v) in nums.iter().enumerate() {
        s1 += i;
        s2 += v;
    }
    s1 as i32 - s2
}

// Two Sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    use std::collections::HashMap;

    let mut m: HashMap<i32, i32> = HashMap::new();
    for (i, v) in nums.iter().enumerate() {
        match m.get(&(target - *v)) {
            Some(&i2) => return vec![i as i32, i2],
            None => m.insert(*v, i as i32),
        };
    }
    vec![]
}

// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Add Two Numbers
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    match (l1, l2) {
        (None, None) => None,
        (Some(n), None) | (None, Some(n)) => Some(n),
        (Some(n1), Some(n2)) => {
            let sum = n1.val + n2.val;
            match sum {
                0..=9 => Some(Box::new(ListNode {
                    val: sum,
                    next: Solution::add_two_numbers(n1.next, n2.next),
                })),
                _ => Some(Box::new(ListNode {
                    val: sum - 10,
                    next: Solution::add_two_numbers(
                        Solution::add_two_numbers(Some(Box::new(ListNode::new(1))), n1.next),
                        n2.next,
                    ),
                })),
            }
        }
    }
}

// Reverse Integer
pub fn reverse(x: i32) -> i32 {
    x.signum()
        * x.abs()
            .to_string()
            .chars()
            .rev()
            .collect::<String>()
            .parse::<i32>()
            .unwrap_or(0)
}

// Longest Substring Without Repeating Characters
pub fn length_of_longest_substring(s: String) -> i32 {
    let mut map = std::collections::HashMap::new();
    let mut start = 0;
    let mut result = 0;
    for (i, c) in s.chars().enumerate() {
        map.entry(c)
            .and_modify(|x| {
                start = start.max(*x + 1);
                *x = i;
            })
            .or_insert(i);
        result = result.max(i - start + 1);
    }
    result as i32
}

// Median of Two Sorted Arrays
pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let mut merged = vec![];
    let (mut i, mut j) = (0, 0);

    while i < nums1.len() && j < nums2.len() {
        if nums1[i] < nums2[j] {
            merged.push(nums1[i]);
            i += 1;
        } else {
            merged.push(nums2[j]);
            j += 1;
        }
    }

    while i < nums1.len() {
        merged.push(nums1[i]);
        i += 1;
    }

    while j < nums2.len() {
        merged.push(nums2[j]);
        j += 1;
    }

    let mid = merged.len() / 2;
    match merged.len() % 2 {
        0 => return ((merged[mid - 1] as f64 + merged[mid] as f64) / 2.0) as f64,
        _ => return merged[mid] as f64,
    }
}

// Longest Valid Parentheses
pub fn longest_valid_parentheses(s: String) -> i32 {
    let mut stack = vec![];
    let mut m = 0;

    stack.push(-1);
    for (i, c) in s.chars().enumerate() {
        match c {
            '(' => stack.push(i as i32),
            _ => {
                stack.pop();
                match stack.is_empty() {
                    true => stack.push(i as i32),
                    false => {
                        let l = i as i32 - stack[stack.len() - 1];
                        m = m.max(l);
                    }
                }
            }
        }
    }
    m
}

// Longest Common Prefix
pub fn longest_common_prefix(strs: Vec<String>) -> String {
    match strs.is_empty() {
        true => "".to_string(),
        _ => strs.iter().skip(1).fold(strs[0].clone(), |acc, x| {
            acc.chars()
                .zip(x.chars())
                .take_while(|(x, y)| x == y)
                .map(|(x, _)| x)
                .collect()
        }),
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Binary Tree Inorder Traversal
pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let (mut res, mut stack) = (vec![], vec![]);
    let mut r = root.clone();

    while r.is_some() || !stack.is_empty() {
        while let Some(node) = r {
            stack.push(node.clone());
            r = node.borrow().left.clone();
        }
        r = stack.pop();
        if let Some(node) = r {
            res.push(node.borrow().val);
            r = node.borrow().right.clone();
        }
    }
    res
}

// Merge Sorted Array
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let (mut m, mut n) = (m as usize, n as usize);
    while m >= 1 && n >= 1 {
        if nums1[m - 1] > nums2[n - 1] {
            nums1[m + n - 1] = nums1[m - 1];
            m -= 1;
        } else {
            nums1[m + n - 1] = nums2[n - 1];
            n -= 1;
        }
    }

    while n >= 1 {
        nums1[m + n - 1] = nums2[n - 1];
        n -= 1;
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Same Tree
pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (p, q) {
        (None, None) => true,
        (Some(p), Some(q)) => {
            let p = p.borrow();
            let q = q.borrow();
            p.val == q.val
                && Self::is_same_tree(p.left.clone(), q.left.clone())
                && Self::is_same_tree(p.right.clone(), q.right.clone())
        }
        _ => false,
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Binary Tree Level Order Traversal
pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut res: Vec<Vec<i32>> = Vec::new();

    fn traversal(root: Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<Vec<i32>>, level: usize) {
        if let Some(r) = root {
            if res.len() == level {
                res.push(Vec::new());
            }
            res[level].push(r.borrow().val);
            traversal(r.borrow().left.clone(), res, level + 1);
            traversal(r.borrow().right.clone(), res, level + 1);
        }
    }
    traversal(root.clone(), &mut res, 0);

    res
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Binary Tree Preorder Traversal
pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    match root {
        Some(r) => {
            let (mut res, mut s) = (vec![], vec![r]);
            while let Some(p) = s.pop() {
                res.push(p.borrow().val);
                if let Some(right) = p.borrow().right.clone() {
                    s.push(right);
                }
                if let Some(left) = p.borrow().left.clone() {
                    s.push(left);
                }
            }
            res
        }
        None => vec![],
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
//
// Binary Tree Postorder Traversal
pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    match root {
        None => vec![],
        Some(root) => {
            let (mut s1, mut s2, mut res) = (vec![root], vec![], vec![]);

            while let Some(n) = s1.pop() {
                if let Some(left) = n.borrow().left.clone() {
                    s1.push(left);
                }
                if let Some(right) = n.borrow().right.clone() {
                    s1.push(right);
                }
                s2.push(n);
            }

            while let Some(n) = s2.pop() {
                res.push(n.borrow().val);
            }
            res
        }
    }
}

// Edit Distance
pub fn min_distance(word1: String, word2: String) -> i32 {
    let (l1, l2) = (word1.len(), word2.len());
    let (w1, w2) = (word1.into_bytes(), word2.into_bytes());

    let mut dp = vec![vec![0; l2 + 1]; l1 + 1];

    for i in 0..=l1 {
        dp[i][0] = i as i32;
    }

    for j in 0..=l2 {
        dp[0][j] = j as i32;
    }

    for i in 1..=l1 {
        for j in 1..=l2 {
            if w1[i - 1] == w2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]) + 1;
            }
        }
    }
    dp[l1][l2]
}

// Remove Duplicates from Sorted Array
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    match nums.is_empty() {
        true => 0,
        false => {
            let mut prev = 0;
            for i in 1..nums.len() {
                if nums[prev] != nums[i] {
                    prev += 1;
                    nums[prev] = nums[i];
                }
            }
            (prev + 1) as i32
        }
    }
}

// Pascal's Triangle
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let mut res = vec![];
    for i in 0..num_rows as usize {
        res.push(vec![1; i + 1]);
        for j in 1..i {
            res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
        }
    }
    res
}

// Pascal's Triangle II
pub fn get_row(row_index: i32) -> Vec<i32> {
    let mut res: Vec<Vec<i32>> = vec![];
    for i in 0..=row_index as usize {
        let mut row: Vec<i32> = vec![1; i + 1];
        for j in 1..i {
            row[j] = res[i - 1][j - 1] + res[i - 1][j];
        }
        if i == row_index as usize {
            return row;
        }
        res.push(row);
    }
    vec![]
}

// Word Break
pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
    let mut v = vec![false; s.len() + 1];
    v[0] = true;
    for i in 0..v.len() {
        match v[i] {
            false => continue,
            true => {
                for word in word_dict.iter() {
                    if let Some(s2) = s.get(i..i + word.len()) {
                        if s2 == *word {
                            v[i + word.len()] = true;
                        }
                    }
                }
            }
        }
    }
    v[s.len()]
}

// Count Primes
pub fn count_primes(n: i32) -> i32 {
    let mut prime = vec![true; n as usize];
    let mut res = 0;

    for i in 2..n as usize {
        if !prime[i] {
            continue;
        }

        res += 1;
        let mut mul = i * i;
        while mul < n as usize {
            prime[mul] = false;
            mul += i;
        }
    }

    res
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Kth Smallest Element in a BST
pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
    fn traversal(n: Option<Rc<RefCell<TreeNode>>>, tmp: &mut Vec<i32>) {
        match n {
            Some(n) => {
                traversal(n.borrow().left.clone(), tmp);
                // Kth Smallest Element in a BST
                tmp.push(n.borrow().val);
                traversal(n.borrow().right.clone(), tmp);
            }
            None => (),
        }
    }

    let mut tmp = vec![];
    traversal(root.clone(), &mut tmp);
    tmp[k as usize - 1]
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Count Complete Tree Nodes
pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn countn(n: Option<Rc<RefCell<TreeNode>>>, c: &mut i32) {
        match n {
            Some(n) => {
                *c += 1;
                countn(n.borrow().left.clone(), c);
                countn(n.borrow().right.clone(), c);
            }
            _ => (),
        }
    }
    let res = &mut 0i32;
    countn(root.clone(), res);
    *res
}

// Kth Largest Element in an Array
pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
    fn qSort(nums: &mut [i32]) {
        if nums.len() <= 1 {
            return;
        }

        let mid = nums.len() / 2;
        let (mut left, right) = (0, nums.len() - 1);
        nums.swap(mid, right);

        let mut i = 0;
        while i < nums.len() {
            if nums[i] < nums[right] {
                nums.swap(i, left);
                left += 1;
            }
            i += 1;
        }

        nums.swap(left, right);
        qSort(&mut nums[0..left]);
        qSort(&mut nums[left + 1..=right]);
    }

    let mut nums = nums;
    qSort(&mut nums);
    nums[nums.len() - k as usize]
}

// Combination Sum III
pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
    fn comb(k: usize, n: i32, mut cur: i32, l: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if l.len() == k {
            return;
        }
        while cur <= 9 {
            let mut c = l.clone();
            let s = c.iter().sum::<i32>() + cur;

            if n < s {
                break;
            }

            c.push(cur);
            if s == n && c.len() == k {
                res.push(c.clone());
                break;
            }

            comb(k, n, cur + 1, &mut c, res);
            cur += 1;
        }
    }

    let (mut l, mut res) = (vec![], vec![]);
    comb(k as usize, n, 1, &mut l, &mut res);
    res
}

// Nim Game
pub fn can_win_nim(n: i32) -> bool {
    n & 3 != 0
}

// Power Of Three
pub fn is_power_of_three(n: i32) -> bool {
    let mut n = n;
    if n == 0 {
        false
    } else {
        while n % 3 == 0 {
            n /= 3
        }
        n == 1
    }
}

// Power of Four
pub fn is_power_of_four(num: i32) -> bool {
    match num <= 0 {
        true => false,
        _ => {
            let mut num = num;
            while num % 4 == 0 {
                num /= 4;
            }
            num == 1
        }
    }
}

// Sum of Two Integers
pub fn get_sum(a: i32, b: i32) -> i32 {
    fn sum(a: i32, b: i32) -> i32 {
        match b {
            0 => a,
            _ => sum(a ^ b, (a & b) << 1),
        }
    }
    sum(a, b)
}

// Find All Duplicates in an Array
pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let mut rtn = vec![];

    for i in 0..nums.len() {
        let n = nums[i].abs();
        let idx = (n - 1) as usize;
        if nums[idx] < 0 {
            rtn.push(n);
        } else {
            nums[idx] = -1 * nums[idx];
        }
    }
    return rtn;
}

// Hamming Distance
pub fn hamming_distance(x: i32, y: i32) -> i32 {
    (x ^ y).count_ones() as i32
}

// Sort Characters By Frequency
pub fn frequency_sort(s: String) -> String {
    let mut counter = std::collections::HashMap::new();
    for c in s.chars() {
        let count = counter.entry(c).or_insert(0);
        *count += 1;
    }
    let mut list = counter.iter().collect::<Vec<(&char, &i32)>>();
    list.sort_unstable_by(|a, b| b.1.cmp(a.1));

    let mut res = vec![];
    for (&ch, &count) in list {
        for _ in 0..count {
            res.push(ch);
        }
    }
    res.iter().collect::<String>()
}

// Minimum Moves to Equal Array Elements
pub fn min_moves(nums: Vec<i32>) -> i32 {
    match nums.iter().min() {
        Some(m) => {
            let mut res = 0;
            for n in &nums {
                res += n - m;
            }
            res
        }
        _ => 0,
    }
}

// Ugly Number
pub fn is_ugly(num: i32) -> bool {
    if num < 1 {
        return false;
    }

    let facts = [2, 3, 5];
    let mut num = num;
    for v in facts.iter() {
        while num % v == 0 {
            num /= v;
        }
    }
    num == 1
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Find Mode in Binary Search Tree
pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    match root {
        None => vec![],
        Some(root) => {
            let mut counter = std::collections::HashMap::<i32, i32>::new();
            fn find(n: Option<Rc<RefCell<TreeNode>>>, m: &mut std::collections::HashMap<i32, i32>) {
                match n {
                    Some(n) => {
                        (*m.entry(n.borrow().val).or_insert(0)) += 1;
                        find(n.borrow().left.clone(), m);
                        find(n.borrow().right.clone(), m);
                    }
                    None => (),
                }
            }
            find(Some(root), &mut counter);
            let mut list = counter.iter().collect::<Vec<(&i32, &i32)>>();
            list.sort_unstable_by(|a, b| b.1.cmp(a.1));

            let (mut res, mut prev) = (vec![*list[0].0], list[0].1);
            for i in 1..list.len() {
                if prev != list[i].1 {
                    break;
                }
                res.push(*(list[i].0));
            }
            res
        }
    }
}

// Base 7
pub fn convert_to_base7(num: i32) -> String {
    let mut num = num;
    match num {
        0 => "0".to_string(),
        _ => {
            let sign = if num < 0 {
                num *= -1;
                "-".to_string()
            } else {
                "".to_string()
            };

            let mut s = String::new();
            while num > 0 {
                s = (num % 7).to_string() + &s;
                num /= 7;
            }
            sign + &s
        }
    }
}

// Relative Ranks
pub fn find_relative_ranks(nums: Vec<i32>) -> Vec<String> {
    let (mut ns, mut m) = (nums.clone(), std::collections::HashMap::new());
    ns.sort_by_key(|&n| -n);
    ns.into_iter().enumerate().for_each(|(i, n)| {
        m.insert(n, i);
    });

    nums.iter()
        .map(|n| match m[n] {
            0 => "Gold Medal".to_string(),
            1 => "Silver Medal".to_string(),
            2 => "Bronze Medal".to_string(),
            _ => (m[n] + 1).to_string(),
        })
        .collect()
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Find Bottom Left Tree Value
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn find_bottom_left_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn find(root: Option<Rc<RefCell<TreeNode>>>, height: i32, max: &mut i32, res: &mut i32) {
            match root {
                Some(root) => {
                    if height > *max {
                        *max = height;
                        *res = root.borrow().val;
                    }
                    find(root.borrow().left.clone(), height + 1, max, res);
                    find(root.borrow().right.clone(), height + 1, max, res);
                }
                _ => (),
            }
        }
        let (max, res) = (&mut 0, &mut 0);
        find(root.clone(), 1, max, res);
        *res
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Most Frequent Subtree Sum
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn find_frequent_tree_sum(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        match root {
            None => vec![],
            Some(root) => {
                let mut m = std::collections::HashMap::new();
                fn count(
                    root: Option<Rc<RefCell<TreeNode>>>,
                    m: &mut std::collections::HashMap<i32, i32>,
                ) -> i32 {
                    match root {
                        Some(root) => {
                            let left = count(root.borrow().left.clone(), m);
                            let right = count(root.borrow().right.clone(), m);
                            let total = left + right + root.borrow().val;
                            (*m.entry(total).or_insert(0)) += 1;
                            total
                        }
                        None => 0,
                    }
                }
                count(Some(root), &mut m);
                let mut list = m.iter().collect::<Vec<(&i32, &i32)>>();
                list.sort_unstable_by(|a, b| b.1.cmp(a.1));

                let (mut res, mut prev) = (vec![*list[0].0], list[0].1);
                for i in 1..list.len() {
                    if prev != list[i].1 {
                        break;
                    }
                    res.push(*list[i].0);
                }
                res

                /*
                or

                m.iter().fold(vec![], |mut acc, (&k, &v)| {
                    if v == max {
                        acc.push(v);
                    }
                    acc
                })
                */
            }
        }
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Find Largest Value in Each Tree Row
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn largest_values(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        fn collect(root: Option<Rc<RefCell<TreeNode>>>, level: usize, ret: &mut Vec<i32>) {
            match root {
                None => (),
                Some(root) => {
                    if ret.len() < level {
                        ret.push(root.borrow().val);
                    } else {
                        ret[level - 1] = ret[level - 1].max(root.borrow().val);
                    }
                    collect(root.borrow().left.clone(), level + 1, ret);
                    collect(root.borrow().right.clone(), level + 1, ret);
                }
            }
        }
        let mut ret = vec![];
        collect(root, 1, &mut ret);
        ret
    }
}

// Remove Element
impl Solution {
    pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
        nums.retain(|&v| v != val);
        nums.len() as i32
    }
}

// Find First and Last Position of Element in Sorted Array
impl Solution {
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        match nums.len() {
            0 => vec![-1, -1],
            _ => {
                let (mut start, mut end) = (-1, -1);
                for i in 0..nums.len() {
                    if nums[i] != target {
                        continue;
                    }

                    if start == -1 {
                        start = i as i32;
                        end = i as i32;
                    } else {
                        end = i as i32;
                    }
                }
                vec![start, end]
            }
        }
    }
}
