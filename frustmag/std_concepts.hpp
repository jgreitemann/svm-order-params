// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018-2019  Jonas Greitemann, Ke Liu, and Lode Pollet

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <iterator>
#include <type_traits>

template <typename T, typename U>
concept bool Same = std::is_same<T, U>::value;

template <typename From, typename To>
concept bool Convertible = std::is_convertible<From, To>::value;

template <typename T>
concept bool Class = std::is_class<T>::value;

template <typename T>
concept bool Pointer = std::is_pointer<T>::value;

template <typename T>
concept bool Integral = std::is_integral<T>::value;

template <typename T>
concept bool Signed = std::is_signed<T>::value;

template <typename T>
concept bool Unsigned = std::is_unsigned<T>::value;

template <typename T>
concept bool UniformRandomBitGenerator = requires {
    typename T::result_type;
    requires Integral<typename T::result_type>;
    requires Unsigned<typename T::result_type>;
    {T::min()} -> typename T::result_type;
    {T::max()} -> typename T::result_type;
    requires requires(T rng) {
        {rng()} -> typename T::result_type;
    };
};

template <typename T>
concept bool DefaultConstructible = std::is_default_constructible<T>::value;

template <typename T>
concept bool CopyConstructible = std::is_copy_constructible<T>::value;

template <typename T>
concept bool CopyAssignable = std::is_copy_assignable<T>::value;

template <typename T>
concept bool Assignable = std::is_assignable<T, T>::value;

template <typename T>
concept bool Destructible = std::is_destructible<T>::value;

template <typename T>
concept bool Swappable = std::is_swappable<T>::value;

template <typename T>
concept bool EqualityComparable = requires(const T a, const T b) {
    {a == b} -> bool;
    {a != b} -> bool;
};

template <typename T>
concept bool Iterator = requires {
    requires CopyConstructible<T>;
    requires CopyAssignable<T>;
    requires Destructible<T>;
    requires Swappable<std::add_lvalue_reference_t<T>>;
    typename std::iterator_traits<T>::value_type;
    typename std::iterator_traits<T>::difference_type;
    typename std::iterator_traits<T>::reference;
    typename std::iterator_traits<T>::pointer;
    typename std::iterator_traits<T>::iterator_category;
    requires requires(T r) {
        {*r};
        {++r} -> std::add_lvalue_reference_t<T>;
    };
};

template <typename T, typename I = typename std::iterator_traits<T>::value_type>
concept bool InputIterator = requires {
    requires Same<I, typename std::iterator_traits<T>::value_type>;
    requires Iterator<T>;
    requires EqualityComparable<T>;
    requires requires(T i) {
        {*i} -> typename std::iterator_traits<T>::reference;
        {*i} -> I;
        {(void)i++};
        {*i++} -> I;
    };
    requires Pointer<T> || requires(T i) {
        {i.operator->()} -> typename std::iterator_traits<T>::pointer;
    };
};

template <typename T, typename O = typename std::iterator_traits<T>::value_type>
concept bool OutputIterator = requires {
    requires Iterator<T>;
    requires Class<T> || Pointer<T>;
    requires requires(std::add_lvalue_reference_t<T> r, O o) {
        {*r = o};
        {++r} -> std::add_lvalue_reference_t<T>;
        {r++} -> std::add_lvalue_reference_t<std::add_const_t<T>>;
        {*r++ = o};
    };
};

template <typename T, typename V = typename std::iterator_traits<T>::value_type>
concept bool ForwardIterator = requires {
    requires InputIterator<T, V>;
    requires DefaultConstructible<T>;
    requires ((OutputIterator<T, V>
               && Same<typename std::iterator_traits<T>::value_type &,
                       typename std::iterator_traits<T>::reference>)
              || Same<typename std::iterator_traits<T>::value_type const&,
                      typename std::iterator_traits<T>::reference>);
    requires requires(T i) {
        {i++} -> T;
        {*i++} -> typename std::iterator_traits<T>::reference;
    };
};

template <typename C>
concept bool Container = requires {
    typename C::value_type;
    typename C::reference;
    typename C::const_reference;
    typename C::iterator;
    typename C::const_iterator;
    typename C::difference_type;
    typename C::size_type;
    // requires Erasable<T>;
    requires Same<typename C::reference,
                  std::add_lvalue_reference_t<typename C::value_type>>;
    requires Same<typename C::const_reference,
                  std::add_lvalue_reference_t<std::add_const_t<typename C::value_type>>>;
    requires ForwardIterator<typename C::iterator, typename C::value_type>;
    requires Convertible<typename C::iterator, typename C::const_iterator>;
    requires ForwardIterator<typename C::const_iterator, typename C::value_type>;
    requires Integral<typename C::difference_type> && Signed<typename C::difference_type>;
    requires Integral<typename C::size_type> && Unsigned<typename C::size_type>;

    requires DefaultConstructible<C>;
    requires CopyConstructible<C>;
    requires Assignable<C>;
    requires EqualityComparable<C>;
    requires Swappable<C>;
    requires requires(C const a) {
        {a.begin()} -> typename C::const_iterator;
        {a.end()} -> typename C::const_iterator;
        {a.size()} -> typename C::size_type;
        {a.max_size()} -> typename C::size_type;
        {a.empty()} -> bool;
    };
    requires requires(C a, C b) {
        {(&a)->~C()} -> void;
        {a.begin()} -> typename C::iterator;
        {a.end()} -> typename C::iterator;
        {a.cbegin()} -> typename C::const_iterator;
        {a.cend()} -> typename C::const_iterator;
        {a.swap(b)} -> void;
    };
};
