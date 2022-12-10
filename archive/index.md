---
layout: page
title: Archive
permalink: /archive/
---

Here I document my experiments, thoughts, and analyses on a variety of topics.
This page also includes my study notes on books I read or courses I follow. I
hope my notebook helps you as much as it has helped me.

<ul>
  {% for post in site.posts %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
        <meta name="description" content="{{ post.summary | escape }}">
        <meta name="keywords" content="{{ post.tags | join: ', ' | escape }}"/>
    </li>
  {% endfor %}
</ul>